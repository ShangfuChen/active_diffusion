"""
Querying mechanisms
"""

import os
import random

def generate_queries(image_directories, query_algorithm, n_queries):
    """
    Given directories of images to use in query, generate queries

    Args:
        image_directories (list(str)) : list of paths to directories where images to be used in queries are located
        query_algorithm (str) : name of query algorithm
        n_queries (int) : number of queries to generate
    """
    QUERY_ALGORITHMS = {
        "random" : _random_query_algorithm,
    }
    assert query_algorithm in QUERY_ALGORITHMS.keys(), f"query_algorithm must be one of {QUERY_ALGORITHMS.keys()}\n Got {query_algorithm}"
    query_function = QUERY_ALGORITHMS[query_algorithm]

    # list out all images (dir/imgx.jpg format)
    img_paths = []

    for img_dir in image_directories:
        img_names = os.listdir(img_dir)
        img_paths += [os.path.join(img_dir, img_name) for img_name in img_names]

    return query_function(img_paths, n_queries)


def _random_query_algorithm(img_paths, n_queries):
    """
    Randomly chooses 2 images
    outputs [
        [img1, img2],       # query 1
        [img3, img2], ...   # query 2
    ]
    """
    queries = []
    for _ in range(n_queries):
        queries.append(random.choices(img_paths, k=2))
    
    return queries
    