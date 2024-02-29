"""
Querying mechanisms
"""

import os
import random
import numpy as np
import torch

class QueryGenerator:
    def __init__(self):
        self.QUERY_ALGORITHMS = {
            "random" : self._random_query_algorithm,
        }

    def generate_queries(self, images, query_algorithm, n_queries):
        """
        Given directories of images to use in query, generate queries

        Args:
            images (list(str) or torch.Tensor) : list of image directories or batch of images in Tensor form (B x C x H x W)
            query_algorithm (str) : name of query algorithm
            n_queries (int) : number of queries to generate
        """
        
        assert query_algorithm in self.QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {self.QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
        query_function = self.QUERY_ALGORITHMS[query_algorithm]

        if type(images[0]) == str:
            # list out all images (dir/imgx.jpg format)
            img_paths = []

            for img_dir in images:
                img_names = os.listdir(img_dir)
                img_paths += [os.path.join(img_dir, img_name) for img_name in img_names]

            return query_function(images=img_paths, n_queries=n_queries), img_paths

        elif type(images) == torch.Tensor:

            return query_function(images=images, n_queries=n_queries)

    # def generate_queries_from_tensor(self, image_batch, query_algorithm, n_queries):
    #     """
    #     Given a batch of images in Tensor form, query algorithm choice and the number of queries,
    #     generate a list of queries.

    #     Args:
    #         image_batch (Tensor) : (B x C x H x W) batch of images
    #         query_algorithm (str) : name of query algorithm
    #         n_queries (int) : number of queries to generate

    #     Returns: 
    #         queries (list(list)) : list of image pairs to be queried 
    #     """

    #     assert query_algorithm in self.QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {self.QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
    #     query_function = self.QUERY_ALGORITHMS[query_algorithm]

    #     return query_function(images=image_batch, n_queries=n_queries)


    # def generate_queries(self, image_directories, query_algorithm, n_queries):
    #     """
    #     Given directories of images to use in query, generate queries

    #     Args:
    #         image_directories (list(str)) : list of paths to directories where images to be used in queries are located
    #         query_algorithm (str) : name of query algorithm
    #         n_queries (int) : number of queries to generate
    #     """
        
    #     assert query_algorithm in self.QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {self.QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
    #     query_function = self.QUERY_ALGORITHMS[query_algorithm]

    #     # list out all images (dir/imgx.jpg format)
    #     img_paths = []

    #     for img_dir in image_directories:
    #         img_names = os.listdir(img_dir)
    #         img_paths += [os.path.join(img_dir, img_name) for img_name in img_names]

    #     return query_function(img_paths, n_queries)


    def _random_query_algorithm(self, images, n_queries):
        """
        Randomly chooses 2 images to query

        Args:
            images (list(str) or tensor) : list of paths to images or a batch of images in the form of tensor (B x C x H x W)

        Returns:
            queries (list(list(int))) : list of image index pairs to query
        """
        queries = []
        if type(images[0]) == str:
            n_images = len(images)
        elif type(images) == torch.Tensor:
            n_images = images.shape[0]

        indices = np.arange(n_images)

        for _ in range(n_queries):
            queries.append(random.choices(indices, k=2))
        
        return queries

    # def _random_query_algorithm(self, img_paths, n_queries):
    #     """
    #     Randomly chooses 2 images
    #     outputs [
    #         [img1, img2],       # query 1
    #         [img3, img2], ...   # query 2
    #     ]
    #     """
    #     queries = []
    #     for _ in range(n_queries):
    #         queries.append(random.choices(img_paths, k=2))
        
    #     return queries
        
