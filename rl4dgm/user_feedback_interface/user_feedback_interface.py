"""
User interface to provide and record human feedback
"""
import os
import datetime
import numpy as np
import math
import random
from enum import IntEnum
from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModel

NUMBERING_FONT_SIZE = 16
NUMBERING_FONT = ImageFont.truetype("FreeMono.ttf", NUMBERING_FONT_SIZE)
PROMPT_FONT_SIZE = 16
PROMPT_FONT = ImageFont.truetype("FreeMono.ttf", PROMPT_FONT_SIZE)

class FeedbackTypes(IntEnum):
    PICK_ONE = 0
    SCORE_ONE = 1
    SCORE_ONE_WRT_BEST = 2
    POSITIVE_INDICES = 3

class OutputTypes(IntEnum):
    BINARY_PREFERENCE = 0
    SCORE = 1
    SCORE_WRT_BEST = 2
    POSITIVE_INDICES = 3

DEFAULT_FEEDBACK_SETTINGS = {
    FeedbackTypes.PICK_ONE : {
        "n_options" : 2,
        "n_feedbacks" : 1,
    },
    FeedbackTypes.SCORE_ONE : {
        "n_options" : 1,
        "n_feedbacks" : 1,
        "score_range" : (1, 5),
    },
    FeedbackTypes.SCORE_ONE_WRT_BEST : {
        "n_options" : 1,
        "n_feedbacks" : 1,
        "score_range" : (-5, 5),
    },
    FeedbackTypes.POSITIVE_INDICES : {
    
    },

}

FEEDBACK_TYPE_TO_ID = {
    "pick-one" : FeedbackTypes.PICK_ONE,
    "score-one" : FeedbackTypes.SCORE_ONE,
    "score-one-wrt-best" : FeedbackTypes.SCORE_ONE_WRT_BEST,
    "positive-indices" : FeedbackTypes.POSITIVE_INDICES,
}

class FeedbackInterface:
    def __init__(
        self,
        feedback_type="positive-indices",
        query_image_size=(2048,1024),
        **kwargs,
    ):
        """
        Args:
            feedback_type (int) : type of feedback to use
            query_image_size (2-tuple of ints) : size of image to display when querying humans
            kwargs : optional arguments for feedback setting
                n_options (int) : number of images to present in a single query (defaults to 2)
                n_feedbacks (int) : number of feedbacks user must provide per query (defaults to 1)
        """
        self.query_image_size = query_image_size
        assert feedback_type in FEEDBACK_TYPE_TO_ID.keys(), f"feedback_type should be one of {FEEDBACK_TYPE_TO_ID.keys()}. Got {feedback_type}."
        self.feedback_type = FEEDBACK_TYPE_TO_ID[feedback_type]
        self.feedback_args = self._initialize_feedback_settings(self.feedback_type, **kwargs)        
        self.preference_function = None
        self.generated_image_size = (256, 256) # size of each image (generated by SD)

        self.is_first_batch = True # flag to do things specific to first time agent is queried (e.g. choose best image)

        # intialize dataframe
        self.reset_dataset()

    def query(self, prompt, img_paths, **kwargs):
        """
        Displays query, get feedback, and add to self.df
        NOTE: Nothing is saved to an external file until save_dataset() is called

        Args:
            img_paths (str or list(str)) : filepath(s) of images to query
            prompt (str) : prompt corresponding to images
        """
        # confirm images and prompt are set
        n_images = len(img_paths)

        expected_n_options = self.feedback_args["n_options"]
        assert n_images == expected_n_options, f"Expected {expected_n_options} images for query. Got {n_images}."

        images = self._process_images(img_paths)
        prompt = self._process_prompts(prompt)

        # generate an image to display to user
        self._save_query_image(images, prompt[0], img_save_path="query_image.png")

        # get user feedback
        feedback = self._get_feedback(prompt=prompt, img_paths=img_paths, **kwargs)

        # append query + feedback to self.df
        self._store_feedback(feedback, img_paths, prompt)

    def query_batch(self, prompts, image_batch, query_indices, **kwargs):
        """
        Version of query() that takes a batch of images in the form of tensor (B x C x H x W),
        and a list of index-pairs to query

        Args:
            prompts (list(str)) : list of prompts corresponding to images in image_batch
            image_batch (Tensor) : (B x C x H x W) tensor of images
            query_indices (list(list(int))) : list of queries where each entry is [idx0, idx1] of images to query
        """
        prompts = self._process_prompts(prompts=prompts)
        feedbacks = []
        for query, prompt in zip(query_indices, prompts):
            # Get query images in PIL format 
            if not isinstance(query, Iterable):
                query = [query]
            images = [to_pil_image(image_batch[idx]) for idx in query]

            # Save query image for user
            self._save_query_image(
                images=images,
                prompt=prompt,
                img_save_path="human_query_image.png",
            )
            # Get feedback
            feedback = self._get_feedback(prompt=prompt, images=images)
            feedbacks.append(feedback)

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=images, prompt=prompt)

        return feedbacks # in case getting values directly is more convenient than saving as datafile
             
    def save_dataset(self, dataset_save_path):
        """
        Save content currently in self.df to .parquet file

        Args:
            dataset_save_path (str) : path to .parquet file to save dataset to (with .parquet extension)
        """
        self.df.to_parquet(dataset_save_path)

    def reset_dataset(self,):
        """
        Clear current content of dataframe
        """
        if self.feedback_args["output_type"] == OutputTypes.BINARY_PREFERENCE:
            # initialize Pick-a-Pic style dataset for "binary preference" output type
            self.df = pd.DataFrame(
                columns=[
                    "are_different", "best_image_uid", "caption", "created_at", "has_label",
                    "image_0_uid", "image_0_url", "image_1_uid", "image_1_url",
                    "jpg_0", "jpg_1",
                    "label_0", "label_1", "model_0", "model_1",
                    "ranking_id", "user_id", "num_example_per_prompt", #"__index_level_0__",
                ],
            )

        elif self.feedback_args["output_type"] == OutputTypes.SCORE:
            # initialize dataset to store image and score for "score" output type
            self.df = pd.DataFrame(
                columns=[
                    "caption", "created_at", "has_label",
                    "jpg_0",
                    "score", "model",
                ],
            )

    def _initialize_feedback_settings(self, feedback_type, **kwargs):
        """
        Given feedback type and kwargs, get feedback settings
        Args:
            feedback_type (int) : one of feedback types in FeedbackTypes
            kwargs (dict) : appropriate arguments for selected feedback type
        """
        feedback_args = {}
        feedback_args["type"] = feedback_type
        if feedback_type == FeedbackTypes.PICK_ONE:
            if "n_options" in kwargs.keys() and kwargs["n_options"] > 1:
               print("Setting n_options to", kwargs["n_options"])
               feedback_args["n_options"] = kwargs["n_options"]
            else:
               print("n_options was not set or was invalid (n_options = [2, 9] for PICK_ONE feedback type). Setting to 2 (default)")
               feedback_args["n_options"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.PICK_ONE]["n_options"]
            feedback_args["valid_options"] = np.arange(feedback_args["n_options"])

            print("n_feedbacks for PICK_ONE feedback type is 1")
            feedback_args["n_feedbacks"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.PICK_ONE]["n_feedbacks"]
            feedback_args["output_type"] = OutputTypes.BINARY_PREFERENCE

        elif feedback_type  == FeedbackTypes.SCORE_ONE:
            feedback_args["n_options"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE]["n_options"]
            feedback_args["n_feedbacks"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE]["n_feedbacks"]
            feedback_args["score_range"] = kwargs["score_range"] if "score_range" in kwargs.keys() else DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE]["score_range"]
            feedback_args["output_type"] = OutputTypes.SCORE
            feedback_args["valid_options"] = np.arange(feedback_args["score_range"][0], feedback_args["score_range"][1]+1)

        elif feedback_type  == FeedbackTypes.SCORE_ONE_WRT_BEST:
            feedback_args["n_options"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE_WRT_BEST]["n_options"]
            feedback_args["n_feedbacks"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE_WRT_BEST]["n_feedbacks"]
            feedback_args["score_range"] = kwargs["score_range"] if "score_range" in kwargs.keys() else DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.SCORE_ONE_WRT_BEST]["score_range"]
            feedback_args["output_type"] = OutputTypes.SCORE_WRT_BEST
            feedback_args["valid_options"] = np.arange(feedback_args["score_range"][0], feedback_args["score_range"][1]+1)

        elif feedback_type == FeedbackTypes.POSITIVE_INDICES:
            feedback_args["output_type"] = OutputTypes.POSITIVE_INDICES
            feedback_args["valid_options"] = None
            feedback_args["min_n_inputs"] = 1 # TODO - magic number. specify through hydra config?

        return feedback_args

    def _process_images(self, img_paths):
        """
        Opens image files to query, store it in a list, and return

        Args:
            img_paths (str or list(str)) : image file name(s) to query

        Returns:
            images (list(PIL.Image)) : list of PIL Images read from img_paths
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        # confirm number of images match feedback type - TODO
        expected_n_images = self.feedback_args["n_options"]
        assert len(img_paths) == expected_n_images, f"Number of input images does not match the feedback settings. Expected {expected_n_images}. Got {len(img_paths)}"

        # [filepaths] -> [images]
        images = []
        for path in img_paths:
            images.append(Image.open(path))

        return images
    
    def _process_prompts(self, prompts):
        """
        Confirms that the prompt is valid.

        Args:
            prompts (str, list(str) or list(tuple(str))) : prompts for the query
        
        Returns:
            prompts (list(str)) : original prompts in a single list 
        """
        if type(prompts) == str:
            prompts = [prompts]
        elif type(prompts[0]) == tuple:
            prompts = [list(tup) for tup in prompts]
            prompts = [prompt for sublist in prompts for prompt in sublist]
        for prompt in prompts:
            assert len(prompt) > 0, "Prompt cannot be an empty string."
        return prompts

    def _save_query_image(self, images, prompt, img_save_path, custom_image_size=None): # TODO
        """
        Given images, prompt, and path to save image, organize it in a grid and number them.
        Save the image to specified path.

        Args:
            images (list of PIL images)
            prompt (str) : text corresponding to query
            img_save_path (str) : file path to save the query image to (with .png extension)
        """
        query_image_size = custom_image_size if custom_image_size is not None else self.query_image_size

        if not isinstance(images, list):
            images = [images]

        # initialize image size
        if self.generated_image_size is None:
            self.generated_image_size = (images[0].size[0], images[0].size[1])
        
        # iif any of the images is None, replace with dummy black image
        dummy_image = Image.new(
            mode="RGB",
            size=(self.generated_image_size[0], self.generated_image_size[1]),
            color="black",
        )
        images = [img if img is not None else dummy_image for img in images]

        # get number of rows and columns
        row_margin = 32 # spacing between each row
        col_margin = 10
        n_images = len(images)

        # get number of rows and columns that maximize the size of each image
        max_size = 0
        n_rows = 0
        n_cols = 0
        for r in range(1, n_images + 1):
            c = math.ceil(n_images / r)
            if c * r >= n_images: # confirm all images can be fit
                size = min(query_image_size[0] / c, query_image_size[1] / r)
                if size > max_size:
                    max_size = size
                    n_rows = r
                    n_cols = c

        # scale images
        scaled_size = int(min(
            (query_image_size[1] - (n_rows + 1) * row_margin) / n_rows, 
            (query_image_size[0] - (n_cols + 1) * col_margin) / n_cols
        ))

        for i in range(len(images)):
            images[i] = images[i].resize((scaled_size, scaled_size))

        # generate image locations (pix position of top left corner of images)
        x_start = int( (query_image_size[0] - n_cols * scaled_size) / 2 )
        x_coords, y_coords = [], []
        for i in range(n_cols):
            x_coords.append( int((i + 1) * col_margin + i * scaled_size) )
            # x_coords.append( int(x_start + i * scaled_size) )
        for i in range(n_rows):
            y_coords.append( int((i + 1) * row_margin + i * scaled_size) )

        # organize images in a grid
        query_image = Image.new(mode="RGB", size=query_image_size, color=(255,255,255))
        coords = []
        for y in range(n_rows):
            for x in range(n_cols):
                coords.append((x_coords[x], y_coords[y]))
        
        draw = ImageDraw.Draw(query_image)
       
        for i, img in enumerate(images):
            query_image.paste(img, coords[i])
            text_pos = (
                coords[i][0] + scaled_size/2 - NUMBERING_FONT_SIZE/2, 
                coords[i][1] + scaled_size + row_margin/2 - NUMBERING_FONT_SIZE/2
            )
            draw.text(text_pos, f"{i+1}", fill=(0,0,0), font=NUMBERING_FONT)
        
        # add prompt
        draw.text((20, 10), text=prompt, fill=(0,0,0), font=PROMPT_FONT)

        # save image
        query_image.save(img_save_path, "PNG")
        print("saved query image")

    def _get_feedback(self,):
        """
        Gets image labels (preference). To be implemented in child classes
        """
        raise NotImplementedError

    def _store_feedback(self, feedback, images, prompt):
        """
        Adds feedback to self.df

        Args:
            images (list(str) or list(PIL.Image)) : list of paths to images or list of PIL Images
            feedback (tuple(float)) : feedback provided by the user (label0, label1)
            prompt (str) : text prompt
        """
        pass
        # if self.feedback_args["output_type"] == OutputTypes.BINARY_PREFERENCE:
        #     img_paths = images.copy()
        #     if not type(images[0]) == str:
        #         img_paths[0] = "tmp_im0.jpg"
        #         img_paths[1] = "tmp_im1.jpg"
        #         images[0].save(img_paths[0])
        #         images[1].save(img_paths[1])

        #     are_different = not img_paths[0] == img_paths[1]
        #     best_image_uid = ""
        #     created_at = datetime.now()
        #     has_label = True
        #     image_0_uid = "0"
        #     image_0_url = ""
        #     image_1_uid = "1"
        #     image_1_url = ""

        #     with open(img_paths[0], "rb") as img0:
        #         jpg_0 = img0.read()

        #     with open(img_paths[1], "rb") as img1:
        #         jpg_1 = img1.read()

        #     # TODO - there is no option for "no preference"
        #     label0, label1 = feedback
        #     model0 = ""
        #     model1 = ""
        #     ranking_id = 0
        #     user_id = 0
        #     num_example_per_prompt = 1
            
        #     self.df.loc[len(self.df.index)] = [
        #         are_different, best_image_uid, prompt, created_at, has_label,
        #         image_0_uid, image_0_url, image_1_uid, image_1_url,
        #         jpg_0, jpg_1,
        #         label0, label1, model0, model1,
        #         ranking_id, user_id, num_example_per_prompt,
        #     ]
        
        # elif self.feedback_args["output_type"] == OutputTypes.SCORE:
        #     img_paths = images.copy()
        #     if not type(images[0]) == str:
        #         img_paths[0] = "tmp_im0.jpg"
        #         images[0].save(img_paths[0])
        #     with open(img_paths[0], "rb") as img0:
        #         jpg_0 = img0.read()
            
        #     caption = prompt
        #     created_at = datetime.now()
        #     has_label = True
        #     score = feedback
        #     model = ""

        #     self.df.loc[len(self.df.index)] = [
        #         caption, created_at, has_label,
        #         jpg_0,
        #         score, model,
        #     ]



class AIFeedbackInterface(FeedbackInterface):
    """
    Auto-generate preference 
    """
    def __init__(
        self,
        preference_function,
        feedback_type=FeedbackTypes.PICK_ONE,
        query_image_size=(1024,1024),
    ):
        """
        Collect preference from AI evaluator

        Args:
            preference_function : function that takes query image paths and prompt as input, and output feedback in the form (label0, label1)
        """
        super().__init__(
            feedback_type=feedback_type,
            query_image_size=query_image_size,
        )
        self.preference_function = preference_function
        if preference_function.__name__ == "PickScore":
            self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.reward_model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").to("cuda").eval()

    def _get_feedback(self, **kwargs):
        """
        Get preference from AI

        Args:
            prompt (str) : prompt
            img_paths (list(str)) : list of paths to images to query
        """
        return self.preference_function(**kwargs)
    
    def query_batch(self, prompts, image_batch, query_indices, **kwargs):
        """
        Version of query() that takes a batch of images in the form of tensor (B x C x H x W),
        and a list of index-pairs to query

        Args:
            prompts (list(str)) : list of prompts corresponding to images in image_batch
            image_batch (Tensor) : (B x C x H x W) tensor of images
            query_indices (list(list(int))) : list of queries where each entry is [idx0, idx1] of images to query
        """
        prompts = self._process_prompts(prompts=prompts)
        feedbacks = []
        for query, prompt in zip(query_indices, prompts):
            # Get query images in PIL format 
            if not isinstance(query, Iterable):
                query = [query]
            images = [to_pil_image(image_batch[idx]) for idx in query]

            # # Save query image for user
            # self._save_query_image(
            #     images=images,
            #     prompt=prompt,
            #     img_save_path="query_image.png",
            # )
            # Get feedback
            if self.preference_function.__name__ == 'PickScore':
                feedback = self._get_feedback(processor=self.processor,
                                              model=self.reward_model,
                                              prompt=prompt,
                                              images=images,
                                              device=image_batch.device,
                                              )
            else:
                feedback = self._get_feedback(prompt=prompt, images=images)
            feedbacks.append(feedback)

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=images, prompt=prompt)

        return feedbacks # in case getting values directly is more convenient than saving as datafile
      
    

class HumanFeedbackInterface(FeedbackInterface):
    """
    Collect preference from a real human evaluator
    """
    def __init__(
        self,
        feedback_type="pick-one",
        query_image_size=(2048,512),
        **kwargs,
    ):
        """
        Args:
            feedback_type (int) : type of feedback to use
            query_image_size (2-tuple of ints) : size of image to display when querying humans
            kwargs : optional arguments for feedback setting
                n_options (int) : number of images to present in a single query (defaults to 2)
                n_feedbacks (int) : number of feedbacks user must provide per query (defaults to 1)
        """
        super().__init__(
            feedback_type=feedback_type,
            query_image_size=query_image_size,
            **kwargs,
        )

        # initialize dictionary to keep track of images most recently labeled with each score
        self.reference_images = {} # imgs are stored as PIL Images
        if self.feedback_type == FeedbackTypes.SCORE_ONE:
            # score_options = np.arange(self.feedback_args["score_range"][0], self.feedback_args["score_range"][1]+1)
            self.reference_images = {score : None for score in self.feedback_args["valid_options"]}

        # keep track of the best image so far
        self.best_score = 0
        self.best_image = None

        # whether to keep track of best image
        self.use_best_image = kwargs["use_best_image"] if "use_best_image" in kwargs.keys() else False

    def _get_feedback(self, valid_options=None, **kwargs):
        if valid_options is None:
            valid_options = self.feedback_args["valid_options"]

        if self.feedback_args["output_type"] == OutputTypes.BINARY_PREFERENCE:
            return self._get_pick_one_labels_feedback(valid_options=valid_options)
        elif self.feedback_args["output_type"] == OutputTypes.SCORE:
            return self._get_raw_feedback(valid_options=valid_options)
        elif self.feedback_args["output_type"] == OutputTypes.SCORE_WRT_BEST:
            return self._get_score_wrt_best(valid_options=valid_options)
        elif self.feedback_args["output_type"] == OutputTypes.POSITIVE_INDICES:
            return self._get_positive_indices_feedback(valid_options=valid_options)

    def _wait_for_feedback(self, valid_options):
        input_str = "-999"

        while True:
            try:
                input_int = int(input_str)
            except: 
                input_str = input(f"Input is not an integer. Enter a number from {valid_options[0]} to {valid_options[-1]}.")
            if input_int in valid_options:
                return input_int
            input_str = input(f"Enter a number from {valid_options[0]} to {valid_options[-1]}.")

    def _get_raw_feedback(self, valid_options):
        # get human input
        input_int = self._wait_for_feedback(valid_options)

        # Get final output
        return input_int

    def _get_pick_one_labels_feedback(self, valid_options):
        # get human input
        input_int = self._wait_for_feedback(valid_options)

        # Get final output 
        if self.feedback_args["output_type"] == OutputTypes.BINARY_PREFERENCE:
            # "binary preference" output type returns labels for images 0 and 1
            if input_int == 1:
                label0 = 1
                label1 = 0
            elif input_int == 2:
                label0 = 0
                label1 = 1
            elif input_int == 0:
                label0 = 0.5
                label1 = 0.5
            return (label0, label1)
        
        else:
            raise Exception("binary preference is the only pick-one feedback type currently supported")

    def _get_score_wrt_best(self, valid_options):
        # get human input
        input_int = self._wait_for_feedback(valid_options)

        # Get final output
        return self.best_score + input_int
    
    def _get_positive_indices_feedback(self, valid_options):
        # get human input (list of positive indices)
        input_str = "none"

        # cast valid indices to list of ints
        if not isinstance(valid_options, list):
            valid_options = valid_options.tolist()
        valid_options = [int(i) for i in valid_options]

        # wait for valid feedback
        while True:
            input_str = input("Enter a list of integers (space-separated) for images comparable or better than the current best image")
            input_indices = input_str.split()

            try: # confirm inputs are a list of ints
                input_ints = [int(idx) for idx in input_indices]
            except:
                print("Got non-integer input(s). Input must be a speace separated list of ints")
                continue

            n_chosen = len(input_ints)
            n_not_chosen = len(valid_options) - len(input_ints)
            if n_chosen < self.feedback_args["min_n_inputs"] or n_not_chosen < self.feedback_args["min_n_inputs"]:
                print(f"Number of chosen and unchosen indices must both be at least {self.feedback_args['min_n_inputs']}. Got chosen ({n_chosen}), not chosen ({n_not_chosen})")
                continue
          
            # make sure all provided indices are in the lit of valid options
            if set(input_ints).issubset(valid_options):
                return np.array(input_ints)
            else:
                print("One or more of the input indices are invalid. Indices must be one of", valid_options)

    def query_batch(self, prompts, image_batch, query_indices, **kwargs):
        """
        Version of query() that takes a batch of images in the form of tensor (B x C x H x W),
        and a list of index-pairs to query

        Args:
            prompts (list(str)) : list of prompts corresponding to images in image_batch
            image_batch (Tensor) : (B x C x H x W) tensor of images
            query_indices (list(list(int))) : list of queries where each entry is [idx0, idx1] of images to query
        """
        if self.feedback_type == FeedbackTypes.SCORE_ONE:
            return self._query_batch_score_one(prompts, image_batch, query_indices, **kwargs)
        elif self.feedback_type == FeedbackTypes.PICK_ONE:
            return self._query_batch_binary_pref(prompts, image_batch, query_indices, **kwargs)
        elif self.feedback_type == FeedbackTypes.SCORE_ONE_WRT_BEST:
            return self._query_batch_score_one_wrt_best(prompts, image_batch, query_indices, **kwargs)
        elif self.feedback_type == FeedbackTypes.POSITIVE_INDICES:
            return self._query_batch_n_pick_m(prompts, image_batch, query_indices, **kwargs)

    def _query_batch_n_pick_m(self, prompts, image_batch, query_indices, **kwargs):
        """
        query batch when feedback type is n pick m
        """
        image_batch = image_batch[query_indices]
        prompts = self._process_prompts(prompts=prompts)
        pil_images = [to_pil_image(image_batch[i]) for i in range(image_batch.shape[0])]
        
        # save query image
        self._save_query_image(
            images=pil_images,
            prompt="Choose the best image",
            img_save_path="real_human_ui_images/query_image.png",
        )
        # if this is the first query, have evaluator choose the first best image
        if self.is_first_batch:
            print("Select the best image in query_image.png")
            best_image_index = self._get_raw_feedback(valid_options=np.arange(1, len(pil_images)+1)) - 1
            self.best_image = pil_images[best_image_index]
     
            # save the best image
            self._save_query_image(
                images=self.best_image,
                prompt="Best image so far",
                img_save_path="real_human_ui_images/best_image.png",
                custom_image_size=(512,512),
            )

        # get feedback
        valid_options = np.arange(1, image_batch.shape[0] + 1)
        positive_indices = self._get_feedback(valid_options=valid_options) - 1
        if self.is_first_batch and best_image_index not in positive_indices:
            positive_indices = np.append(positive_indices, best_image_index)
        negative_indices = np.setdiff1d(valid_options - 1, positive_indices)

        # update best image?
        if not self.is_first_batch:
            best_image_candidates = [pil_images[i] for i in positive_indices] + [self.best_image]
            self._save_query_image(
                images=best_image_candidates,
                prompt="Choose the best image",
                img_save_path="real_human_ui_images/best_image_candidates.png",
                custom_image_size=(1024,1024),
            )
            best_image_index = self._get_raw_feedback(valid_options=np.arange(1, len(best_image_candidates)+1)) - 1
            # update best image if anything other than the current best is chosen
            if not best_image_index == len(best_image_candidates) - 1:
                self.best_image = best_image_candidates[best_image_index]

                # save the best image
                self._save_query_image(
                    images=self.best_image,
                    prompt="Best image so far",
                    img_save_path="real_human_ui_images/best_image.png",
                    custom_image_size=(512,512),
                )
            else:
                # if best image is not updated, don't return index
                best_image_index = None
        self.is_first_batch = False

        return best_image_index, positive_indices, negative_indices


    def _query_batch_binary_pref(self, prompts, image_batch, query_indices, **kwargs):
        """
        query batch when feedback type is binary preference
        """                    
        prompts = self._process_prompts(prompts=prompts)
        feedbacks = []
        for query, prompt in zip(query_indices, prompts):
            # Get query images in PIL format 
            if not isinstance(query, Iterable):
                query = [query]
            images = [to_pil_image(image_batch[idx]) for idx in query]

            # Save query image for user
            self._save_query_image(
                images=images,
                prompt=prompt,
                img_save_path="real_human_ui_images/query_image.png",
            )
            # Get feedback
            feedback = self._get_feedback(prompt=prompt, images=images)
            feedbacks.append(feedback)

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=images, prompt=prompt)

        return feedbacks # in case getting values directly is more convenient than saving as datafile
    
    def _query_batch_score_one(self, prompts, image_batch, query_indices, **kwargs):
        prompts = self._process_prompts(prompts=prompts)
        feedbacks = []
        pil_images = []

        for query, prompt in zip(query_indices, prompts):
            # Get query images in PIL format 
            if not isinstance(query, Iterable):
                query = [query]
            images = [to_pil_image(image_batch[idx]) for idx in query]

            # Save query image for user
            self._save_query_image(
                images=images,
                prompt=prompt,
                img_save_path="real_human_ui_images/query_image.png",
            )

            # save reference image (most recent image given each score)
            self._save_query_image(
                images=list(self.reference_images.values()),
                prompt="Reference images",
                img_save_path="real_human_ui_images/reference_image.png",
            )
            pil_images += images

            # Get feedback
            print("Best score so far is: ", self.best_score)
            feedback = self._get_feedback(prompt=prompt, images=images)
            feedbacks.append(feedback)

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=images, prompt=prompt)

            # update reference image and best score 
            self.reference_images[feedback] = images[0]
            # update best score so far
            if feedback > self.best_score:
                self.best_score = feedback

        if self.use_best_image:
            # get images with best score so far and save ui image
            best_score_indices = np.where(np.array(feedbacks) == self.best_score)[0]
            best_images = [pil_images[i] for i in best_score_indices]
            if len(best_images) == 1:
                # if there is only one candidate, no need to query human
                self.best_image = best_images[0]
            else:
                # otherwise, save query image to let human choose which one is best
                if self.best_image is not None:
                    # if best image has already been assigned, include this to the best image candidates
                    best_images += [self.best_image]
                self._save_query_image(
                    images=best_images,
                    prompt="Best image candidates",
                    img_save_path="real_human_ui_images/best_image_candidates.png",
                )
                best_image_index = self._get_raw_feedback(valid_options=np.arange(len(best_images))+1)
                self.best_image = best_images[best_image_index - 1]

            # save the best image so far
            self._save_query_image(
                images=self.best_image,
                prompt="Best image so far",
                img_save_path="real_human_ui_images/best_image.png",
            )

        return feedbacks # in case getting values directly is more convenient than saving as datafile

    def _query_batch_score_one_wrt_best(self, prompts, image_batch, query_indices, **kwargs):
        prompts = self._process_prompts(prompts=prompts)
        feedbacks = []
        pil_images = []

        # if this is the first query, have evaluator choose the first best image
        if self.is_first_batch:
            print("Select the best image in best_image_candidates.png")
            pil_images = [to_pil_image(image_batch[idx]) for idx in query_indices]
            self._save_query_image(
                images=pil_images,
                prompt="Choose the best image",
                img_save_path="real_human_ui_images/best_image_candidates.png",
            )
            best_image_index = self._get_raw_feedback(valid_options=np.arange(1, len(pil_images)+1)) - 1
            self.best_image = pil_images[best_image_index]
            self.best_score = 5 # TODO - magic number

            # save the best image so far
            self._save_query_image(
                images=self.best_image,
                prompt="Best image so far",
                img_save_path="real_human_ui_images/best_image.png",
            )
        
        print("Start collecting feedback")
        for query, prompt in zip(query_indices, prompts):
            print("query", query)
            # Get query images in PIL format 
            if not isinstance(query, Iterable):
                query = [query]
            images = [to_pil_image(image_batch[idx]) for idx in query]
            # Save query image for user
            self._save_query_image(
                images=images,
                prompt=prompt,
                img_save_path="real_human_ui_images/query_image.png",
            )

            # TODO - this mode currently does not support reference image saving
            # save reference image (most recent image given each score)
            # self._save_query_image(
            #     images=list(self.reference_images.values()),
            #     prompt="Reference images",
            #     img_save_path="real_human_ui_images/reference_image.png",
            # )
            pil_images += images

            # Get feedback
            if self.is_first_batch and query[0] == best_image_index:
                feedback = 5 # TODO - magic number
                self.best_score = feedback 
            else:
                print("Best score so far is: ", self.best_score)
                feedback = self._get_feedback(prompt=prompt, images=images)
            
            feedbacks.append(feedback)

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=images, prompt=prompt)

            # update reference image and best score 
            # self.reference_images[feedback] = images[0]
            # update best score so far
            if feedback > self.best_score:
                self.best_score = feedback

        if self.use_best_image and not self.is_first_batch:
            print("Choose the best image in best_image_candidates.png")
            # get images with best score so far and save ui image
            best_score_indices = np.where(np.array(feedbacks) == self.best_score)[0]
            best_images = [pil_images[i] for i in best_score_indices]
            if self.best_image is not None:
                best_images += [self.best_image]
            if len(best_images) == 1:
                # if there is only one candidate, no need to query human
                self.best_image = best_images[0]
            else:
                # otherwise, save query image to let human choose which one is best
                self._save_query_image(
                    images=best_images,
                    prompt="Best image candidates",
                    img_save_path="real_human_ui_images/best_image_candidates.png",
                )
                best_image_index = self._get_raw_feedback(valid_options=np.arange(len(best_images))+1)
                self.best_image = best_images[best_image_index - 1]

            # save the best image so far
            self._save_query_image(
                images=self.best_image,
                prompt="Best image so far",
                img_save_path="real_human_ui_images/best_image.png",
            )

        self.is_first_batch = False
        return feedbacks # in case getting values directly is more convenient than saving as datafile

