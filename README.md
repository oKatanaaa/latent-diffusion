# Latenet Diffusion Model

This fork contains fixed dependencies as well as ready-to-use docker file. The repository was adapted so that it can work on both Linux and Windows machines.

A jupyter notebook has been included with an example of running a super-resolution model.

For the project documentation please check the original repo.


### How to use

#### Docker

Run the following:

1. `git clone https://github.com/oKatanaaa/latent-diffusion.git`
2. `cd latent-diffusion`
3. `docker build -t ldm .`
4. `docker run -it -d --gpus all --name ldm ldm`

After that attach VS Code to the container and you are good to go.

#### Outside docker

Requirements:
- conda (don't forget conda init)

Run the following:

1. `git clone https://github.com/oKatanaaa/latent-diffusion.git`
2. `cd latent-diffusion`
3. `conda env create -f ./environment.yaml`
4. `conda activate ldm`
5. `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

The environment is setup and ready to use.

## Precautions

This repository was tested on RTX 2060 with CUDA 11.6. Depending on your CUDA version, you may want to change `environment.yml` a bit (cudatoolkit version).
