from cgi import test
import os
from glob import glob
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from notebook_helpers import get_model, run, get_cond_for_psnr_testing


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



def preprocess(im_pair):
    # Related to how the model processes large images.
    # There will be black borders (correspond to edge patches with dims < 256px)
    assert im_pair[0].shape == im_pair[1].shape
    new_h = (im_pair[0].shape[0] // 256) * 256
    new_w = (im_pair[0].shape[1] // 256) * 256
    return im_pair[0][:new_h, :new_w], im_pair[1][:new_h, :new_w]


def estimate_psnr(image_filenames, model, diff_steps, save_path=None):
    psnr_vals = []
    ssim_vals = []
    for image_file in image_filenames:
        example = get_cond_for_psnr_testing('superresolution', image_file)
        logs = run(model, example, 'superresolution', diff_steps)
        
        orig_image = extract_orig(example)
        test_image = extract_sample(logs)
        orig_image, test_image = preprocess((orig_image, test_image))
        psnr = peak_signal_noise_ratio(orig_image, test_image)
        ssim = structural_similarity(orig_image, test_image, channel_axis=2)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        
        print(f'PSNR: {psnr:.2f}, SSIM: {ssim:.2f}, im_path: {image_file}')
        if save_path is not None:
            image_name = image_file.split(os.sep)[-1]
            savepath = os.path.join(save_path, image_name)
            test_image = Image.fromarray(test_image)
            test_image.save(savepath)
    return np.mean(psnr_vals), np.mean(ssim_vals)


def main(dataset_path, save_path):
    model = get_model('superresolution')
    image_filenames = glob(os.path.join(dataset_path, '*'))
    psnr, ssim = estimate_psnr(image_filenames, model, DIFFUSION_STEPS, save_path)
    print(f'Metrics: mean psnr={psnr:.2f}, mean ssim={ssim:.2f}')
    
    with open(os.path.join(save_path, 'psnr.txt'), 'w') as f:
        f.write(f'Metrics: mean psnr={psnr:.2f}, mean ssim={ssim:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--save', required=True)
    
    args = parser.parse_args()
    
    print(f'Received: data_path={args.data}, save_path={args.save}')
    main(args.data, args.save)