import sys, os

sys.path.insert(0, os.getcwd())

import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler, EulerDiscreteScheduler
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import numpy as np
import imageio.v3 as imageio
from PIL import Image

from torchvision import transforms

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="outputs")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--model_id', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--n_images', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=512)
    opt = parser.parse_args()

    return opt


@torch.no_grad()
def main(opt):

    preprocess = transforms.Compose(
        [
            transforms.Resize((opt.image_size, opt.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    vae = AutoencoderKL.from_pretrained(opt.model_id, subfolder="vae")
    device = "cuda:0"
    vae.to(device)

    files = [_ for _ in os.listdir(opt.input_dir) if _.endswith(".JPEG")]
    files = sorted(files)[:opt.n_images]
    os.makedirs(opt.output_dir, exist_ok=True)

    for file in tqdm(files):

        image = Image.open(os.path.join(opt.input_dir, file))
        image.save(os.path.join(opt.output_dir, file))
        input = preprocess(image).unsqueeze(0).to(device)

        if input.shape[1] == 1: input = input.repeat(1, 3, 1, 1)

        latents = vae.encode(input).latent_dist.mean

        np.save(os.path.join(opt.output_dir, file[:-5] + ".npy"), latents.cpu().numpy().astype(np.float32))

        # print(latents.shape)
        # decoded = vae.decode(latents).sample
        # print(decoded.shape)

        # decoded = torch.clamp((decoded * 0.5 + 0.5) * 255, 0, 255).cpu().permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        # imageio.imwrite(os.path.join(opt.output_dir, file[:-4] + "_decoded.png"), decoded)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)