
import os
import argparse
import numpy as np

from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

def score_and_save_images(prompt, img_dir, img_save_dir, hf_model_path, ckpt_path=None):    

    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    if ckpt_path is not None:
        # see if model_path is a path to local checkpoint
        model = AutoModel.from_pretrained(ckpt_path).eval().to(device)
        print("model loaded from checkpoint")
    else:
        # load from huggingface
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_path).eval().to(device)

    def calc_probs(prompt, images):
        
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist(), scores.cpu().tolist()

    scores = {}
    imgs = []
    img_names = os.listdir(img_dir)
    for img in img_names:
        imgs.append(Image.open(os.path.join(img_dir, img)))

    probs, scores = calc_probs(images=imgs, prompt=prompt)
    breakpoint()
    # print("list of images\n", img_names)
    # print("scores\n", scores)

    imgs_low2high = [img_name for _, img_name in sorted(zip(scores, img_names), reverse=True)]

    # print("low to high scores\n", imgs_low2high)

    # save images with ranking
    import shutil
    os.makedirs(img_save_dir, exist_ok=True)
    for i, img_name in enumerate(imgs_low2high):
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(img_save_dir, f"rank{i}_{img_name}"))

    min_score = min(scores)
    max_score = max(scores)
    print(f"range of scores: {min_score} to {max_score}")
    print("average score:", np.mean(scores))
    print("stdev", np.std(scores))
    breakpoint()

def main(args):
    score_and_save_images(
        prompt=args.prompt,
        img_dir=args.img_dir,
        img_save_dir=args.img_save_dir,
        hf_model_path=args.hf_model_path,
        ckpt_path=args.ckpt_path,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--img-save-dir", type=str)
    parser.add_argument("--hf-model-path", type=str, default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--ckpt-path", type=str)
    args = parser.parse_args()
    main(args)