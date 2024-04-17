from PIL import Image
import io
import numpy as np
import torch
import torchvision


"""
If loading the original PickScore model:
    model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").eval().to(device)

If loading from checkpoint:
    model = AutoModel.from_pretrained(ckpt_path).eval().to(device)

    where ckpt_path = "checkpoint-final", given the following file structure:
        checkpoint-final
        |_ config.json
        |_ pytorch_model.bin
        |_ training_stage.json
"""
def calc_probs(processor, model, images, prompt, device):
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
        try:
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        except:
            image_embs = model.module.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.module.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.module.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist(), scores.cpu().tolist()


"""
Use finetuned pickscore to calculate rewards
"""
def pickscore():
    def _fn(processer, model, images, prompts, device):
        if isinstance(images, torch.Tensor):
            imgs = []
            toPIL = torchvision.transforms.ToPILImage()
            for im in images:
                imgs.append(toPIL(im))    
        _, score = calc_probs(processer, model, imgs, prompts, device)
        return score, {}
    return _fn


"""
Use pixel values from a channel as a dummy reward
"""
def color_score():
    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
            red = np.mean(images[:, 0, :, :], axis=(1, 2))
            other = np.mean(images[:, 1:, :, :], axis=(1, 2, 3))
            score = 1 + (red - other)*9
            score += np.random.randn(score.shape[0])*1
            # score += 1
        return score, {}
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}
    return _fn

# NOTE: remove metadata args, which is only for llava and is not used here
def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts):
        rew, meta = jpeg_fn(images, prompts)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn
