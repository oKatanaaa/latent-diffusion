import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from notebook_helpers import get_cond_for_psnr_testing, get_model, run

DIFFUSION_STEPS = 100


def extract_sample(logs) -> Image:
    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    return sample[0]


def extract_orig(example):
    orig_image = np.asarray(example['orig'])[0]
    return (orig_image * 255).astype(np.uint8)


def estimate_psnr(image_filenames, model, diff_steps, save_path=None):
    psnr_vals = []
    for image_file in image_filenames:
        example = get_cond_for_psnr_testing('superresolution', image_file)
        logs = run(model, example, 'superresolution', diff_steps)
        
        orig_image = extract_orig(example)
        test_image = extract_sample(logs)
        #return example, orig_image, test_image
        psnr = peak_signal_noise_ratio(orig_image, test_image)
        psnr_vals.append(psnr)
        
        print(f'PSNR: {psnr:.2f}, im_path: {image_file}')
        if save_path is not None:
            image_name = image_file.split(os.sep)[-1]
            savepath = os.path.join(save_path, image_name)
            test_image = Image.fromarray(test_image)
            test_image.save(savepath)
    return np.mean(psnr_vals)


def main(dataset_path, save_path):
    model = get_model('superresolution')
    image_filenames = glob(os.path.join(dataset_path, '*'))
    psnr = estimate_psnr(image_filenames, model, DIFFUSION_STEPS, save_path)
    print(f'Computed mean psnr: {psnr:.2f}')
    
    with open(os.path.join(save_path, 'psnr.txt'), 'w') as f:
        f.write(f'Computed mean psnr: {psnr:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--save', required=True)
    
    args = parser.parse_args()
    
    print(f'Received: data_path={args.data}, save_path={args.save}')
    main(args.data, args.save)