import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
import einops as E
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import center_padding
import cv2

class SAM(nn.Module):
    def __init__(self, return_multilayer=True):
        super().__init__()
        # 加载SAM模型
        model = sam_model_registry["vit_l"](checkpoint="./checkpoints/sam_vit_l_0b3195.pth")
        self.image_encoder = model.image_encoder

        # 获取 patch size
        self.patch_size = self.image_encoder.patch_embed.proj.stride[0]

        # 获取特征维度
        feat_dim = self.image_encoder.blocks[0].mlp.lin2.out_features
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # 获取特征提取的目标层
        n_layers = len(self.image_encoder.blocks)
        multilayers = [
            n_layers // 4 - 1,  # 第5层
            n_layers // 2 - 1,  # 第11层
            n_layers // 4 * 3 - 1,  # 第17层
            n_layers - 1,  # 第23层
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)  # 5-11-17-23

    def forward(self, images):
        # 图像填充
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # 图像编码器的patch embedding
        x = self.image_encoder.patch_embed(images)  # 1*16*16*1024
        # x = E.rearrange(x, "b h w c -> b (h w) c")  # 1*256*1024

        # 提取多层特征
        embeds = []
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x) #1*16*16*1024
            if i in self.multilayers:
                # 将特征图重新排列为 (b, c, h, w) 格式
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        return embeds


def feature_sam():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)    

    # sam = sam_model_registry["vit_l"](checkpoint="./checkpoints/sam_vit_l_0b3195.pth").to(device='cuda')
    # generator = SamAutomaticMaskGenerator(sam)
    # image = cv2.imread('/data/winter25/shenm/ProRobo3D/feature_extract/front_rgb.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pri
    # masks = generator.generate(image)

    model = SAM().to('cuda')
    # model = nn.DataParallel(model, device_ids=[1, 2, 3, 4, 5, 6, 7]) 
    # # with open('model.txt', 'w') as f:
    # #     print(model, file=f)

    img_dim_resized = (256*8, 256*8)
    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    image = Image.open(img_dir).convert('RGB')
    image = transform(image).unsqueeze(0).to('cuda')
    # print(image.shape) #1, 3, 256, 256

    # 提取特征
    with torch.no_grad():
        feat_2d = model(image) 
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # print(feat_2d[0].shape) # 输出形状: (1, 16, 16, 1024)

    feat_2d = torch.cat(feat_2d[-3:], dim=3)
    # print(feat_2d.shape) #[1, 16, 16, 3072]
    feat_2d = feat_2d.squeeze(0).permute(2, 0, 1)
    # print(feat_2d.shape) #[3072, 16, 16]
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    # print(feat_2d.shape) #[3072, 256, 256]

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 3072)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('sam_ord.png', feat_map_pca_reshaped)

if __name__ == "__main__":
    feature_sam()