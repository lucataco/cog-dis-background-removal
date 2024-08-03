# predict.py
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from cog import BasePredictor, Input, Path
# project imports
from data_loader_cache import normalize, im_reader, im_preprocess 
from models import *

class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        return normalize(image, self.mean, self.std)

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set up model parameters
        self.hypar = {
            "model_path": "./checkpoints",
            "restore_model": "isnet.pth",
            "interm_sup": False,
            "model_digit": "full",
            "seed": 0,
            "cache_size": [1024, 1024],
            "input_size": [1024, 1024],
            "crop_size": [1024, 1024],
            "model": ISNetDIS()
        }
        
        self.transform = transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])
        self.net = self.build_model()

    def build_model(self):
        net = self.hypar["model"]
        if self.hypar["model_digit"] == "half":
            net.half()
            for layer in net.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
        net.to(self.device)
        net.load_state_dict(torch.load(f"{self.hypar['model_path']}/{self.hypar['restore_model']}", 
                                       map_location=self.device, weights_only=True))
        net.eval()
        return net

    def load_image(self, im_path):
        im = im_reader(im_path)
        im, im_shp = im_preprocess(im, self.hypar["cache_size"])
        im = torch.divide(im, 255.0)
        shape = torch.from_numpy(np.array(im_shp))
        return self.transform(im).unsqueeze(0), shape.unsqueeze(0)

    def predict(self, image: Path = Input(description="Input image for segmentation")) -> list[Path]:
        image_tensor, orig_size = self.load_image(str(image))
        
        if self.hypar["model_digit"] == "full":
            image_tensor = image_tensor.type(torch.FloatTensor)
        else:
            image_tensor = image_tensor.type(torch.HalfTensor)

        inputs_val_v = image_tensor.to(self.device)
        ds_val = self.net(inputs_val_v)[0]
        pred_val = ds_val[0][0,:,:,:]

        pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val,0),
                                               size=(orig_size[0][0],orig_size[0][1]),
                                               mode='bilinear',
                                               align_corners=False))

        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val-mi)/(ma-mi)

        mask = (pred_val.detach().cpu().numpy()*255).astype(np.uint8)
        
        pil_mask = Image.fromarray(mask).convert('L')
        im_rgb = Image.open(image).convert("RGB")
        
        im_rgba = im_rgb.copy()
        im_rgba.putalpha(pil_mask)

        output_path = Path(f"/tmp/output.png")
        mask_path = Path(f"/tmp/mask.png")
        im_rgba.save(output_path)
        pil_mask.save(mask_path)

        return [output_path, mask_path]