import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16

def feature_spa():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    image = Image.open(img_dir).convert('RGB')
    img_dim_resized = (256, 256) #图像缩放
    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    image = transform(image).unsqueeze(0).to('cuda')
    # print(image.shape) #1, 3, 252, 252

    model = spa_vit_large_patch16(pretrained=True)
    model.eval()
    model.freeze()
    model = model.cuda()

    # feat_2d = model(image, feature_map=True, cat_cls=True)  # torch.Size([1, 2048, 16, 16])足够
    feat_2d = model(image, feature_map=True, cat_cls=False)  # torch.Size([1, 1024, 16, 16])
    print(feat_2d.shape)

    feat_2d = feat_2d.squeeze(0)
    print(feat_2d.shape) #1024, 16, 16
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
    print(feat_2d.shape) #1024, 256, 256

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 1024)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('spa.png', feat_map_pca_reshaped)

if __name__ == "__main__":
    feature_spa()