import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import einops as E
from diffusers import StableDiffusionPipeline

class StableDiffusionFeatureExtractor(nn.Module):
    def __init__(self, model_name="sd-legacy/stable-diffusion-v1-5", return_multilayer=True):
        super().__init__()
        # 加载 Stable Diffusion Pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.feature_extractor = self.pipe.feature_extractor  # 通常用于图像预处理
        
        # 设置提取的目标层
        self.return_multilayer = return_multilayer
        
        # U-Net 的输入层特征维度
        self.feat_dim = self.unet.config.in_channels
        self.layer = "unet"  # 统一层标识，表示我们要提取 UNet 特征

    def forward(self, images):
        # 将图像预处理为适合模型的输入格式
        processed_images = self.feature_extractor(images, return_tensors="pt", do_rescale=False).pixel_values.to("cuda")

        # 通过 UNet 提取图像特征
        # UNet 使用额外的条件（如文本条件），但这里仅提取图像的特征
        # 通过 UNet 的前向传播获取特征图
        hidden_dim = self.unet.config.cross_attention_dim 
        encoder_hidden_states = torch.zeros(processed_images.shape[0], 77, hidden_dim).to("cuda")  # 模拟文本输入（可选）
        latents = self.vae.encode(processed_images).latent_dist.sample().to("cuda")  # 使用 VAE 获取潜在空间表示

        # 获取 UNet 的特征图
        output = self.unet(latents, timestep=100, encoder_hidden_states=encoder_hidden_states).sample

        # 根据返回设置提取层
        if self.return_multilayer:
            # 返回所有层的特征
            return output  # 在实际中，如果需要获取中间层，可能需要访问 UNet 的具体层
        
        return output

class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        model_name="sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae

    def forward(self, image, timestep=100):
        # 保存中间层特征
        intermediate_features = []

        self.feature_extractor = self.pipe.feature_extractor
        hidden_dim = self.unet.config.cross_attention_dim 
        processed_images = self.feature_extractor(image, return_tensors="pt", do_rescale=False).pixel_values.to("cuda")
        self.encoder_hidden_states = torch.zeros(processed_images.shape[0], 77, hidden_dim).to("cuda")  # 模拟文本输入（可选）
        self.latents = self.vae.encode(processed_images).latent_dist.sample().to("cuda")  # 使用 VAE 获取潜在空间表示

        # 前向传播通过 down_blocks
        x = self.latents
        # print(x.shape) #1, 4, 28, 28
        x = self.unet.conv_in(x)
        # print(x.shape) #1, 320, 28, 28
        x = E.rearrange(x, "b c h w -> b c (h w)")
        # print(x.shape) #1, 320, 784
        # x = self.unet.time_embedding(x)
        # print(x.shape) #1, 784, 1280

        for down_block in self.unet.down_blocks:
            x = down_block(x, encoder_hidden_states=self.encoder_hidden_states)
            intermediate_features.append(x)  # 保存每一层的输出
            print(x.shape)

        # 前向传播通过 mid_block
        x = self.unet.mid_block(x, encoder_hidden_states=self.encoder_hidden_states)
        print(x.shape)

        # 前向传播通过 up_blocks
        for up_block in self.unet.up_blocks:
            x = up_block(x, encoder_hidden_states=self.encoder_hidden_states)
            intermediate_features.append(x)  # 保存每一层的输出
            print(x.shape)

        # 返回最终输出和中间层特征
        return x, intermediate_features

def feature_sd():

    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    img_dim_resized = (256, 256)

    # model_id = "sd-legacy/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # with open('model.txt', 'w') as f:
    #     print(pipe.unet, file=f)

    transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                ]
            )
    
    img_dir = '/data/winter25/shenm/ProRobo3D/feature_extract/visualization/front_rgb.png'
    image = Image.open(img_dir).convert('RGB')
    image = transform(image).unsqueeze(0).to('cuda')
    print(image.shape) #1, 3, 256, 256

    model = StableDiffusionFeatureExtractor()
    with torch.no_grad():
        feat_2d = model(image)
        print(feat_2d.shape) #1, 4, 28, 28

    # custom_unet = CustomUNet()
    # output, intermediate_features = custom_unet(image=image)
    # print(len(intermediate_features))
    
    feat_2d = feat_2d.squeeze(0)
    print(feat_2d.shape) #[4, 28, 28]
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0) 
    print(feat_2d.shape) #[4, 256, 256]

    feat_map_reshaped = feat_2d.cpu().numpy().transpose(1, 2, 0).reshape(-1, 4)
    pca = PCA(n_components=3)
    feat_map_pca = pca.fit_transform(feat_map_reshaped)
    feat_map_pca_norm = (feat_map_pca - feat_map_pca.min(axis=0)) / (feat_map_pca.max(axis=0) - feat_map_pca.min(axis=0))
    feat_map_pca_reshaped = feat_map_pca_norm.reshape(256, 256, 3)
    plt.imsave('sd1.png', feat_map_pca_reshaped)


if __name__ == "__main__":
    feature_sd()