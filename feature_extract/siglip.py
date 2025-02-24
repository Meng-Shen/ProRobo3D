import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import open_clip
from torch import nn
import einops as E
from utils import center_padding

class SigLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-L-14",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        self.checkpoint_name = "siglip_" + arch.replace("-", "").lower() + checkpoint

        # Initialize a pre-trained SigLIP image encoder and freeze it.
        _siglip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=checkpoint
        )
        _siglip_model = _siglip_model.eval().to(torch.float32)
        self.visual = _siglip_model.visual.trunk  # SigLIP 的视觉部分是 `trunk`
        del _siglip_model

        # Extract some attributes from SigLIP module for easy access.
        self.patch_size = self.visual.patch_embed.proj.stride[0]  # 获取 patch size

        # Get feature dimension
        feat_dim = self.visual.blocks[0].mlp.fc2.out_features  # 获取特征维度
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]

        # Get extraction targets
        n_layers = len(self.visual.blocks)  # SigLIP 的 Transformer 层数
        multilayers = [
            n_layers // 4 - 1, #5
            n_layers // 2 - 1, #11
            n_layers // 4 * 3 - 1, #17
            n_layers - 1, #23
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers) #5-11-17-23

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # SigLIP stuff
        x = self.visual.patch_embed.proj(images)  # 1*1024*16*16
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c") #1*256*1024

        embeds = []
        for i, blk in enumerate(self.visual.blocks):  # SigLIP 的 Transformer 块
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break
        return embeds

def feature_siglip():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # print(open_clip.list_pretrained())
    model = SigLIP(arch='ViT-L-16-SigLIP-256', checkpoint='webli', output="dense", layer=-1, return_multilayer=True).cuda()

    img_dim_resized = (256*16, 256*16)
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
        # print(feat_2d[0].shape) # 输出形状: (1, 256, 1024)

    feat_2d = torch.cat(feat_2d[-3:], dim=2)
    # print(feat_2d.shape) #[1, 256, 3072]
    feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, 16*16, 16*16)
    # print(feat_2d.shape) #[3072, 16, 16]
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    # print(feat_2d.shape) #[3072, 256, 256]

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 3072)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('siglip_ord.png', feat_map_pca_reshaped)


if __name__ == "__main__":
    feature_siglip()