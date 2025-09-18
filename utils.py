# =============================================================================
# Import required libraries
# =============================================================================
import time
import cv2

import torch
from facenet_pytorch import MTCNN

from assets.face_recognition_models import facenet


class MyTimer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


mtcnn = MTCNN(image_size=512,  # Size of the input image
              margin=0,
              post_process=False,
              select_largest=False,
              device='cuda')


def alignment(image):
    boxes, probs = mtcnn.detect(image)
    if boxes is not None: 
        return boxes[0]
    else:
        None


def load_FR_model(args):
    FR_model = {}
    FR_model['facenet'] = []
    FR_model['facenet'].append((160, 160))
    fr_model = facenet.InceptionResnetV1(
        num_classes=8631, device=args.device)
    fr_model.load_state_dict(torch.load(
        'assets/face_recognition_models/facenet.pth'))
    fr_model.to(args.device)
    fr_model.eval()
    FR_model['facenet'].append(fr_model)          
    return FR_model


def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)
    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im


def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img
