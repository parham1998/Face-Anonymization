from .networks import get_model
import torch
import torch.nn.functional as F


class FaceParsing(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.model = get_model('FaceParseNet50', pretrained=False).to(device)
        self.model.load_state_dict(torch.load(
            'assets/face_parsing/38_G.pth', map_location='cpu'))
        self.model.eval()

    def forward(self, x):
        # (B, 3, 512, 512)
        outputs = self.model(x)[0][-1]
        imsize = x.shape[2]
        inputs = F.interpolate(input=outputs, size=(
            imsize, imsize), mode='bilinear', align_corners=True)

        pred_batch = torch.argmax(inputs, dim=1)

        return pred_batch
