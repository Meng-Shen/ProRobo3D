import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
import timm

def feature_uni3d():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

    # img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    # image = Image.open(img_dir).convert('RGB')
    # img_dim_resized = (256, 256) #图像缩放
    # transform = transforms.Compose(
    #             [
    #                 transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
    #                 transforms.ToTensor(),
    #             ]
    #         )
    # image = transform(image).unsqueeze(0).to('cuda')
    # # print(image.shape) #1, 3, 256, 256

    # if [ "$1" = "giant" ]; then
    #     pc_model="eva_giant_patch14_560"
    #     pc_feat_dim=1408
    # elif [ "$1" = "large" ]; then
    #     pc_model="eva02_large_patch14_448"
    #     pc_feat_dim=1024
    # elif [ "$1" = "base" ]; then
    #     pc_model="eva02_base_patch14_448"
    #     pc_feat_dim=768
    # elif [ "$1" = "small" ]; then
    #     pc_model="eva02_small_patch14_224"
    #     pc_feat_dim=384
    # elif [ "$1" = "tiny" ]; then
    #     pc_model="eva02_tiny_patch14_224"
    #     pc_feat_dim=192
    # else
    #     echo "Invalid option"
    #     exit 1
    # fi
    point_transformer = timm.create_model("eva_giant_patch14_560", checkpoint_path='/data/winter25/shenm/ProRobo3D/feature_extract/checkpoints/uni3d.pt')
    point_encoder = PointcloudEncoder(point_transformer, args)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    with open('model.txt', 'w') as f:
        print(model, file=f)
    with torch.no_grad():
        feat_2d = evaluator.forward_features(image)["x_norm_patchtokens"]
        # print(feat_2d.shape) #1, 18*18, 1024

if __name__ == "__main__":
    feature_uni3d()