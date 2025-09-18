# 	Controllable Localized Face Anonymization via Diffusion Inpainting [[ArXiv]()] 

<p align="center">
 <a href="https://parham1998.github.io/" target="_blank">Ali Salar</a>,
 <a href="https://sites.google.com/site/qingliucs/home" target="_blank">Qing Liu</a>,
 <a href="https://gyzhao-nm.github.io/Guoying/" target="_blank">Guoying Zhao</a>
 <br>
</p>

## Abstract
<p align="justify"> The growing use of portrait images in computer vision highlights the need to protect personal identities. At the same time, anonymized images must remain useful for downstream computer vision tasks. In this work, we propose a unified framework that leverages the inpainting ability of latent diffusion models to generate realistic anonymized images. Unlike prior approaches, we have complete control over the anonymization process by designing an adaptive attribute-guidance module that applies gradient correction during the reverse denoising process, aligning the facial attributes of the generated image with those of the synthesized target image. Our framework also supports localized anonymization, allowing users to specify which facial regions are left unchanged. Extensive experiments conducted on the public CelebA-HQ and FFHQ datasets show that our method outperforms state-of-the-art approaches while requiring no additional model training. </p>

## Setup
- **Get code**
```shell 
git clone https://github.com/parham1998/Face-Anonymization.git
```

- **Build environment**
```shell
cd Face-Anonymization
# use anaconda to build environment 
conda create -n Face-Anonymization python=3.11.7
conda activate Face-Anonymization
# install packages
pip install -r requirements.txt
```

- **Download assets and place them in the assets folder**
  - Download datasets from [Datasets](https://drive.google.com/drive/folders/1D87bLfBm6PEvdi7DV2mahaIqUWtScmCb?usp=sharing)
  - Download pre-trained face parsing model from [Face_Parsing](https://github.com/TracelessLe/FaceParsing.PyTorch)
  - Download pre-trained facenet model from [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN)
  - Download pre-trained FaRL from [FaRL](https://github.com/FacePerceiver/FaRL)
  - Download pre-trained LDM from [LDM](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt)

- **The final assets folder should be like this:**
```shell
assets
  └- datasets
    └- CelebA-HQ
    └- FFHQ
  └- face_parsing
    └- 38_G.pth
  └- face_recognition_models
    └- facenet.pth
    └- facenet.py
  └- farl
    └- FaRL-Base-Patch16-LAIONFace20M-ep16.pth
    └- FaRL-Base-Patch16-LAIONFace20M-ep64.pth
  └- ldm
    └- 512-inpainting-ema.ckpt
  └- target_images
```

- **[Datasets](https://drive.google.com/drive/folders/1D87bLfBm6PEvdi7DV2mahaIqUWtScmCb?usp=sharing) are already aligned. However, for new data, the images should be aligned before starting the anonymization process:**
```shell
python align.py
```

- **For anonymization:**
```shell
source_dir=source images folder path
target_path=the desired synthesized target image path
MTCNN_cropping=True
excluded_masks=choose number from: {'2: nose', '3: eye_glasses', '4: l_eye', '5: r_eye', '6: l_brow', '7:r_brow', '10: mouth', '11: u_lip', '12: l_lip'}
```

5. Run the code:
```shell
python main.py
```

## Citation 
```bibtex
```

## Acknowledgments
Our code structure is based on [stablediffusion](https://github.com/Stability-AI/stablediffusion/tree/main)
