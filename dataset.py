# =============================================================================
# Import required libraries
# =============================================================================
import os
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images


class ImageDataset(Dataset):
    def __init__(self,
                 source_dir,
                 target_path,
                 transform=None):
        self.source_dir = sorted(make_dataset(source_dir))
        self.target_path = target_path
        self.transform = transform

    def __len__(self):
        return len(self.source_dir)

    def __getitem__(self, index):
        fname, from_path = self.source_dir[index]
        image = Image.open(from_path).convert('RGB')
        #
        tgt_image = Image.open(self.target_path).convert('RGB')
        #
        if self.transform:
            image = self.transform(image)
            tgt_image = self.transform(tgt_image)

        return fname, image, tgt_image
