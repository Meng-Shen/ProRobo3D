import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from video_diffusion.dift_sd import SDFeaturizer
import open_clip
from torch import nn
import torch.nn.functional as F
import sys
import einops as E
from utils import center_padding, resize_pos_embed, tokens_to_output

def feature_sd():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    img_dim_resized = (256*8, 256*8)
    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    image = Image.open(img_dir).convert('RGB')
    image = transform(image).unsqueeze(0).unsqueeze(0).to('cuda')
    image = image * 2 - 1
    # print(image.shape) #[1, 1, 3, 512, 512]

    model = SDFeaturizer()
    evaluator = model
    with torch.no_grad():
        feat_2d = evaluator.forward(image, t=100, up_ft_index=[0,1,2]) 
        # print(feat_2d[0].shape) #[1, 1280, 16, 16]
        # print(feat_2d[1].shape) #[1, 1280, 32, 32]
        # print(feat_2d[2].shape) #[1, 640, 64, 64]
    
    # resize the feat_2d from 60x80 to 240x320
    feat_2d_0 = torch.nn.functional.interpolate(feat_2d[0], size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    feat_2d_1 = torch.nn.functional.interpolate(feat_2d[1], size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    feat_2d_2 = torch.nn.functional.interpolate(feat_2d[2], size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    feat_2d = torch.cat([feat_2d_0, feat_2d_1, feat_2d_2], dim=0) # 3200, 256, 256

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 3200)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('sd.png', feat_map_pca_reshaped)


if __name__ == "__main__":
    feature_sd()