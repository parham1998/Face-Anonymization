import pathlib
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance

torch.manual_seed(1)
fid = FrechetInceptionDistance(normalize=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def load_images_from_folder(folder, img_format):
    image_paths = pathlib.Path(folder).glob("*."+img_format+"*")
    images = [transform(Image.open(img_path).convert('RGB'))
              for img_path in image_paths]
    images = torch.stack(images)
    return images


real_images = load_images_from_folder('assets/datasets/FFHQ/all', 'png')
fake_images = load_images_from_folder('results/images', 'png')

fid.update(real_images, real=True)
fid.update(fake_images, real=False)
print(fid.compute())
