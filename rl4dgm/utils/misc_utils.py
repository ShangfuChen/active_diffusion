
import os
import shutil

def save_images_in_single_dir(source_img_dirs, save_dir):
    """
    Given a list of image directories, save everything in these directories to a new location (save_dir)
    Images are re-numbered arbitrarily

    Args:
        source_img_dirs (list(str)) : list of image directories to combine
        save_dir (str) : destination of all images
    """

    os.makedirs(save_dir)
    n_saved_imgs = 0
    for source_dir in source_img_dirs:
        paths = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
        for path in paths:
            shutil.copy(path, os.path.join(save_dir, f"{n_saved_imgs}.jpg"))
            n_saved_imgs += 1

source_img_dirs = [
    "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch0",
    "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch1",
    "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch2",
    "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch3",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch4",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch5",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch6",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch7",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch8",
    # "/home/hayano/sampled_images/2024.04.04_21.51.32/epoch9",

]

save_images_in_single_dir(source_img_dirs=source_img_dirs, save_dir="/home/hayano/test")