# active-diffusion

This is an implementation of active query for diffusion models building on [Denoising Diffusion Policy Optimization (DDPO)](https://rl-diffusion.github.io/) in PyTorch with support for [low-rank adaptation (LoRA)](https://huggingface.co/docs/diffusers/training/lora).
If LoRA is enabled, requires less than 10GB of GPU memory to finetune Stable Diffusion!

## Installation
### DDPO
Requires Python 3.10
```bash
cd ddpo
pip install -e .
```
### rl4dgm
```bash
cd rl4dgm/rl4dgm
pip install -e .
```
### Pickscore
```bash
cd Pickscore
pip install -e .
```


## Usage
```bash
accelerate launch scripts/train.py --config config/base.py
```
This will immediately start finetuning Stable Diffusion v1.5 for compressibility on all available GPUs using the config from `config/base.py`. It should work as long as each GPU has at least 10GB of memory. If you don't want to log into wandb, you can run `wandb disabled` before the above command.

### prompt_fn and reward_fn
At a high level, the problem of finetuning a diffusion model is defined by 2 things: a set of prompts to generate images, and a reward function to evaluate those images. The prompts are defined by a `prompt_fn` which takes no arguments and generates a random prompt each time it is called. The reward function is defined by a `reward_fn` which takes in a batch of images and returns a batch of rewards for those images. All of the prompt and reward functions currently implemented can be found in `ddpo/prompts.py` and `ddpo/rewards.py`, respectively.
