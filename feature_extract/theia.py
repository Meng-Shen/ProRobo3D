import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModel

def feature_theia():

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
    image = transform(image).permute(1, 2, 0).unsqueeze(0).to('cuda')
    # print(image.shape) #1, 256, 256, 3

    model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", trust_remote_code=True).to('cuda')

    with torch.no_grad():
        feat_2d = model.forward_feature(image, do_rescale=False)
        # print(feat_2d.shape) #1, 196, 768

    feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, 14, 14) #图像缩放
    # print(feat_2d.shape) #768, 14, 14
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
    # print(feat_2d.shape) #768, 256, 256

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 768)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('theia.png', feat_map_pca_reshaped)

if __name__ == "__main__":
    feature_theia()
