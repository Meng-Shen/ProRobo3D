import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn

def feature_dinov2():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

    # DINOv2
    # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

    # # DINOv2 with registers
    # dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    # dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    # dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    # dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    evaluator = model

    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    image = Image.open(img_dir).convert('RGB')

    img_dim_resized = (252*7, 252*7) #图像缩放
    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    image = transform(image).unsqueeze(0).to('cuda')
    # print(image.shape) #1, 3, 252, 252

    with torch.no_grad():
        feat_2d = evaluator.forward_features(image)["x_norm_patchtokens"]
        # print(feat_2d.shape) #1, 18*18, 1024

    feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, 18*7, 18*7) #图像缩放
    # print(feat_2d.shape) #1024, 18, 18
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
    # print(feat_2d.shape) #1024, 240, 320

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 1024)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('dino.png', feat_map_pca_reshaped)

if __name__ == "__main__":
    feature_dinov2()
