"""
Notes on how to decode pickapic byte image data
"""

from datasets import load_dataset
from datasets.features.features import Value
from PIL import Image
from io import BytesIO
import pandas as pd
import csv
import argparse


# # Read custom dataset
# data_path = "dummy_dataset.csv"
# pd_data = pd.read_csv(data_path)
# pd_dict_data = pd_data.to_dict()

# with open(data_path) as file:
#     reader = csv.DictReader(file)
#     csv_dict_data = [row for row in reader]

# # Load small pickapic dataset
# dataset = load_dataset("xzuyn/pickapic_v2_only_some")

# # grab a sample
# train_dict = dataset["train"].to_dict()
# jpg_0 = train_dict["jpg_0"]
# sample = jpg_0[0]

# breakpoint()

# # read and save image
# img_save_path = "test.jpg"
# im = Image.open(BytesIO(sample))
# im.save(img_save_path, "JPEG")

# breakpoint()


###### Compare Datatypes ######
def check_data(datafile_path):
    # my dataset
    # parquet_path = "my_dataset/my_dataset_validation_unique.parquet"
    mydataset = load_dataset("parquet", data_files={"train" : datafile_path})
    # mytrainset = mydataset["train"]
    # myfeatures = mytrainset.features

    # pickapic dataset
    dataset = load_dataset("xzuyn/pickapic_v2_only_some")
    trainset = dataset["train"]
    features = trainset.features

    # for key in features:
    #     print("\nEntry: ", key)
    #     print("Match 1: ", type(features[key]) == type(myfeatures[key]))
    #     if type(features[key]) == Value:
    #         print("Match 2: ", features[key].dtype == myfeatures[key].dtype)

    for split in mydataset.keys():
        print("="*50)
        print("Split: ", split)
        print("number of rows:", mydataset[split].num_rows)
        myfeatures = mydataset[split].features
        n_mismatches = 0
        for key in features:
            if not type(features[key]) == type(myfeatures[key]):
                print(f"Mismatched datatype for {key}. Expected {type(features[key])}. Got {type(myfeatures[key])}")
                if type(features[key]) == Value:
                    if not features[key].dtype == myfeatures[key].dtype:
                        print(f"Mismatched datatype for {key}. Expected {features[key].dtype}. Got {myfeatures[key].dtype}.")
                    n_mismatches += 1
        print(f"{n_mismatches} datatype mismatches detected")

    breakpoint()

def main(args):
    check_data(args.datafile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, help="path to datafile to test")
    args = parser.parse_args()
    main(args)