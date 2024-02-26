"""
User interface to provide and record human feedback
"""
import os
import datetime

import numpy as np
import random
from enum import IntEnum
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image

NUMBERING_FONT_SIZE = 64
NUMBERING_FONT = ImageFont.truetype("FreeMono.ttf", NUMBERING_FONT_SIZE)
PROMPT_FONT_SIZE = 32
PROMPT_FONT = ImageFont.truetype("FreeMono.ttf", PROMPT_FONT_SIZE)

class FeedbackTypes(IntEnum):
    PICK_ONE = 0

DEFAULT_FEEDBACK_SETTINGS = {
    FeedbackTypes.PICK_ONE : {
        "n_options" : 2,
        "n_feedbacks" : 1,
    },
}

class FeedbackInterface:
    def __init__(
        self,
        feedback_type=FeedbackTypes.PICK_ONE,
        query_image_size=(1024,1024),
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
        self.feedback_type = feedback_type
        self.feedback_args = self._initialize_feedback_settings(feedback_type, **kwargs)        

        # intialize dataframe (Pick-a-Pick style)
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

        # TODO - only supports binary preference feedback typefor now
        assert n_images == 2, f"Expected 2 images for query. Got {n_images}."

        images = self._process_images(img_paths)
        prompt = self._process_prompt(prompt)

        # generate an image to display to user
        self._save_query_image(images, prompt, img_save_path="query_image.png")

        # get user feedback
        feedback = self._get_feedback(prompt=prompt, img_paths=img_paths, **kwargs)

        # append query + feedback to self.df
        self._store_feedback(feedback, img_paths, prompt)

    def query_batch(self, prompt, image_batch, query_indices, **kwargs):
        """
        Version of query() that takes a batch of images in the form of tensor (B x C x H x W),
        and a list of index-pairs to query

        Args:
            prompt (str) : prompt
            image_batch (Tensor) : (B x C x H x W) tensor of images
            query_indices (list(list(int))) : list of queries where each entry is [idx0, idx1] of images to query
        """
        prompt = self._process_prompt(prompt)

        for query in query_indices:
            # Get query images in PIL format 
            idx0, idx1 = query
            im0 = to_pil_image(image_batch[idx0])
            im1 = to_pil_image(image_batch[idx1])

            # Save query image for user
            self._save_query_image(
                images=[im0, im1],
                prompt=prompt,
                img_save_path="query_image.png",
            )

            # Get feedback
            feedback = self._get_feedback(prompt=prompt, images=[im0, im1])

            # Append query + feedback to self.df
            self._store_feedback(feedback=feedback, images=[im0, im1], prompt=prompt)
        

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
        self.df = pd.DataFrame(
            columns=[
                "are_different", "best_image_uid", "caption", "created_at", "has_label",
                "image_0_uid", "image_0_url", "image_1_uid", "image_1_url",
                "jpg_0", "jpg_1",
                "label_0", "label_1", "model_0", "model_1",
                "ranking_id", "user_id", "num_example_per_prompt", #"__index_level_0__",
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
        if feedback_type == FeedbackTypes.PICK_ONE:
            if "n_options" in kwargs.keys() and kwargs["n_options"] > 1:
               print("Setting n_options to", kwargs["n_options"])
               feedback_args["n_options"] = kwargs["n_options"]
            else:
               print("n_options was not set or was invalid (n_options = [2, 9] for PICK_ONE feedback type). Setting to 2 (default)")
               feedback_args["n_options"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.PICK_ONE]["n_options"]

            print("n_feedbacks for PICK_ONE feedback type is 1")
            feedback_args["n_feedbacks"] = DEFAULT_FEEDBACK_SETTINGS[FeedbackTypes.PICK_ONE]["n_feedbacks"]

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
    
    def _process_prompt(self, prompt):
        """
        Confirms that the prompt is valid.

        Args:
            prompt (str) : prompt for the query
        
        Returns:
            prompt (str) : original prompt
        """
        assert len(prompt) > 0, "Prompt cannot be an empty string."
        return prompt

    def _save_query_image(self, images, prompt, img_save_path): # TODO
        """
        Given images, prompt, and path to save image, organize it in a grid and number them.
        Save the image to specified path.

        Args:
            images (list of PIL images)
            prompt (str) : text corresponding to query
            img_save_path (str) : file path to save the query image to (with .png extension)
        """
        # get number of rows and columns
        row_margin = 64 # spacing between each row
        n_images = len(images)
        n_cols = int(min(n_images, 3))
        if n_images % n_cols == 0:
          n_rows = int(n_images / n_cols)
        else:
          n_rows = int((n_images // n_cols) + 1)

        # scale images
        img_size = images[0].size[0]
        scaled_size = int(min(
          (self.query_image_size[1] - (n_rows + 1) * row_margin) / n_rows, 
          self.query_image_size[0] / n_cols)
        )
        for i in range(len(images)):
          images[i] = images[i].resize((scaled_size, scaled_size))

        # generate image locations (pix position of top left corner of images)
        x_start = int( (self.query_image_size[0] - n_cols * scaled_size) / 2 )
        x_coords, y_coords = [], []
        for i in range(n_cols):
          x_coords.append( int(x_start + i * scaled_size) )
        for i in range(n_rows):
          y_coords.append( int((i + 1) * row_margin + i * scaled_size) )

        # organize images in a grid
        query_image = Image.new(mode="RGB", size=self.query_image_size, color=(255,255,255))
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

        img_paths = images.copy()
        if not type(images[0]) == str:
            img_paths[0] = "tmp_im0.jpg"
            img_paths[1] = "tmp_im1.jpg"
            images[0].save(img_paths[0])
            images[1].save(img_paths[1])

        # TODO currently only supports binary feedback preference
        are_different = not img_paths[0] == img_paths[1]
        best_image_uid = ""
        created_at = datetime.now()
        has_label = True
        image_0_uid = "0"
        image_0_url = ""
        image_1_uid = "1"
        image_1_url = ""

        with open(img_paths[0], "rb") as img0:
            jpg_0 = img0.read()

        with open(img_paths[1], "rb") as img1:
            jpg_1 = img1.read()

        # TODO - there is no option for "no preference"
        label0, label1 = feedback
        model0 = ""
        model1 = ""
        ranking_id = 0
        user_id = 0
        num_example_per_prompt = 1
        
        self.df.loc[len(self.df.index)] = [
            are_different, best_image_uid, prompt, created_at, has_label,
            image_0_uid, image_0_url, image_1_uid, image_1_url,
            jpg_0, jpg_1,
            label0, label1, model0, model1,
            ranking_id, user_id, num_example_per_prompt,
        ]
    

    # def _store_feedback(self, feedback, img_paths, prompt):
    #     """
    #     Adds feedback to self.df

    #     Args:
    #         img_paths (list(str)) : list of paths to images included in this query
    #         feedback (tuple(float)) : feedback provided by the user (label0, label1)
    #         prompt (str) : text prompt
    #     """

    #     # TODO currently only supports binary feedback preference
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

    def _get_feedback(self, **kwargs):
        """
        Get preference from AI

        Args:
            prompt (str) : prompt
            img_paths (list(str)) : list of paths to images to query
        """
        return self.preference_function(**kwargs)

class HumanFeedbackInterface(FeedbackInterface):
    """
    Collect preference from a real human evaluator
    """
    def __init__(
        self,
        feedback_type=FeedbackTypes.PICK_ONE,
        query_image_size=(1024,1024),
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

    def _get_feedback(self, **kwargs):
        """
        Prompts the user to input feedback

        Args:
            n_options (int) : number of valid options for the user (choices are numbered from 1 to n_options) 

        Returns:
            feedback (tuple(float)) : (label0, label1) indicating which image was choosen by user
        """
        n_options = 2 # TODO - hardcoded for now (only allow preference feedback)
        input_str = "-1"
        valid_options = np.arange(1, n_options + 1).tolist()
        while int(input_str) not in valid_options:
            input_str = input(f"Enter a number from 1 to {n_options} corresponding to the image that best matches the prompt.\n")

        if input_str == "1":
            label0 = 1
            label1 = 0
        elif input_str == "2":
            label0 = 0
            label1 = 1
        elif input_str == "0":
            label0 = 0.5
            label1 = 0.5

        return (label0, label1)

