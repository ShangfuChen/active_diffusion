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

NUMBERING_FONT_SIZE = 64
NUMBERING_FONT = ImageFont.truetype("FreeMono.ttf", NUMBERING_FONT_SIZE)
PROMPT_FONT_SIZE = 32
PROMPT_FONT = ImageFont.truetype("FreeMono.ttf", PROMPT_FONT_SIZE)

class UserFeedbackTypes(IntEnum):
    PICK_ONE = 0

DEFAULT_FEEDBACK_SETTINGS = {
    UserFeedbackTypes.PICK_ONE : {
        "n_options" : 2,
        "n_feedbacks" : 1,
    },
}

class UserFeedbackInterface:
    def __init__(
        self,
        feedback_type=UserFeedbackTypes.PICK_ONE,
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

    def query(self, image_paths, prompt):
        """
        Displays query, get feedback, and add to self.df
        NOTE: Nothing is saved to an external file until save_dataset() is called

        Args:
            image_paths (str or list(str)) : filepath(s) of images to query
            prompt (str) : prompt corresponding to images
        """
        # confirm images and prompt are set
        n_images = len(image_paths)

        # TODO - only supports binary preference feedback typefor now
        assert n_images == 2, f"Expected 2 images for query. Got {n_images}."

        images = self._process_images(image_paths)
        prompt = self._process_prompt(prompt)

        # generate an image to display to user
        self._save_query_image(images, prompt, img_save_path="query_image.png")

        # get user feedback
        feedback = self._get_feedback(n_options=n_images)

        # append query + feedback to self.df
        self._store_feedback(feedback, image_paths, prompt)

        # # save feedback to file
        # self._save_feedback(feedback, image_paths, prompt, self.dataset_save_path)

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
            feedback_type (int) : one of feedback types in UserFeedbackTypes
            kwargs (dict) : appropriate arguments for selected feedback type
        """
        feedback_args = {}
        if feedback_type == UserFeedbackTypes.PICK_ONE:
            if "n_options" in kwargs.keys() and kwargs["n_options"] > 1:
               print("Setting n_options to", kwargs["n_options"])
               feedback_args["n_options"] = kwargs["n_options"]
            else:
               print("n_options was not set or was invalid (n_options = [2, 9] for PICK_ONE feedback type). Setting to 2 (default)")
               feedback_args["n_options"] = DEFAULT_FEEDBACK_SETTINGS[UserFeedbackTypes.PICK_ONE]["n_options"]

            print("n_feedbacks for PICK_ONE feedback type is 1")
            feedback_args["n_feedbacks"] = DEFAULT_FEEDBACK_SETTINGS[UserFeedbackTypes.PICK_ONE]["n_feedbacks"]

        return feedback_args

    def _process_images(self, image_paths):
        """
        Opens image files to query, store it in a list, and return

        Args:
            image_paths (str or list(str)) : image file name(s) to query

        Returns:
            images (list(PIL.Image)) : list of PIL Images read from image_paths
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # confirm number of images match feedback type - TODO
        expected_n_images = self.feedback_args["n_options"]
        assert len(image_paths) == expected_n_images, f"Number of input images does not match the feedback settings. Expected {expected_n_images}. Got {len(image_paths)}"

        # [filepaths] -> [images]
        images = []
        for path in image_paths:
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

    def _get_feedback(self, n_options):
        """
        Prompts the user to input feedback

        Args:
            n_options (int) : number of valid options for the user (choices are numbered from 1 to n_options) 
        
        Returns:
            feedback (str) : valid feedback entered by user
        """

        feedback = "0"
        valid_options = np.arange(1, n_options + 1).tolist()
        while int(feedback) not in valid_options:
            feedback = input(f"Enter a number from 1 to {n_options} corresponding to the image that best matches the prompt.\n")

        return feedback

    def _store_feedback(self, feedback, image_paths, prompt):
        """
        Adds a human feedback to self.df

        Args:
            image_paths (list(str)) : list of paths to images included in this query
            feedback (str) : feedback provided by the user
            prompt (str) : text prompt
        """

        # TODO currently only supports binary feedback preference
        are_different = not image_paths[0] == image_paths[1]
        best_image_uid = ""
        created_at = datetime.now()
        has_label = True
        image_0_uid = "0"
        image_0_url = ""
        image_1_uid = "1"
        image_1_url = ""
        
        with open(image_paths[0], "rb") as img0:
            jpg_0 = img0.read()

        with open(image_paths[1], "rb") as img1:
            jpg_1 = img1.read()
            
        # TODO - there is no option for "no preference"
        print("feedback", feedback)
        label0 = 1 if feedback == "1" else 0
        label1 = 1 if feedback == "2" else 0
        print("label0, label1", label0, label1)
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



# image_paths = []
# img_source_path = "/home/hayano/prompt_test/Premium_soft_serve_ice_cream."
# img_save_path = "query_img.png"

# n_options = 2
# for i in range(1, n_options+1):
#   image_paths.append(os.path.join(img_source_path, f"{i}.jpg"))

# prompt = "Premium ice cream"

# print(f"{len(image_paths)} images given as input")
# print("image_paths", image_paths)
# kwargs = {
#    "n_options" : n_options,
# }
# ui = UserFeedbackInterface(feedback_type=UserFeedbackTypes.PICK_ONE, **kwargs)
# for i in range(3):
#   random.shuffle(image_paths)
#   ui.query(image_paths, prompt)

#   breakpoint()
# breakpoint()

