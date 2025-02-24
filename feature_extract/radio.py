import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

def feature_radio():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    image = Image.open(img_dir).convert('RGB')
    img_dim_resized = (256*16, 256*16)
    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    image = transform(image).unsqueeze(0).to('cuda')
    print(image.shape) #1, 3, 252, 252

    #model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
    model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
    # model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
    #model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
    #model_version="e-radio_v2" # for E-RADIO
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True, trust_repo=True)
    # model = torch.hub.load('/data/winter25/shenm/.cache/torch/hub/NVlabs_RADIO_main', 'radio_model', version=model_version, progress=True, skip_validation=True)
    model.cuda().eval()
    if "e-radio" in model_version:
        model.model.set_optimal_window_size(x.shape[2:])

    nearest_res = model.get_nearest_supported_resolution(*image.shape[-2:])
    x = F.interpolate(image, nearest_res, mode='bilinear', align_corners=False)

    with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        summary, feat_2d = model(x, feature_fmt='NCHW')
        assert feat_2d.ndim == 4
        # print(feat_2d.shape) #1, 1280, 16, 16

    feat_2d = feat_2d.squeeze(0)
    # print(feat_2d.shape) #1280, 16, 16
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
    # print(feat_2d.shape) #1280, 256, 256

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 1280)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('radio.png', feat_map_pca_reshaped)

if __name__ == "__main__":
    feature_radio()
