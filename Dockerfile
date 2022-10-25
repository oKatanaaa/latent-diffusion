FROM conda/miniconda3

RUN apt-get update && conda update conda && apt-get install git -y

COPY . /latent-diffusion
WORKDIR /latent-diffusion

RUN conda env create -f ./environment.yaml
RUN conda run -n ldm pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda run -n ldm pip install ipykernel

RUN echo "conda activate ldm" > ~/.bashrc
ENV PATH /opt/conda/envs/ldm/bin:$PATH
CMD /bin/bash