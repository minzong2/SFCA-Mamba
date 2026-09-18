
import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from utils import blc_2_bchw, bchw_2_blc
from .PatchEmbed import PatchEmbed
from .sample import DownSample, UpSample
from .RefinementBlock import RefinementBlock  
from .FreqMambaParallelUnit import FreqMambaParallelUnit

def make_light_conv(dim):
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
        nn.BatchNorm2d(dim),
        nn.GELU(),
        nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    )


class PCSformer(nn.Module):
    def __init__(self, config):
        super(PCSformer, self).__init__()
        model_resolution = config['model']['model_resolution']
        in_channels = config['model']['in_channels']
        embed_dim = config['model']['embed_dim']              	depth = config['model']['depth']                  
        split_size = config['model']['split_size']
        num_heads = config['model']['num_heads']
        mlp_ratio = config['model']['mlp_ratio']
        qkv_bias = config['model']['qkv_bias']
        attn_drop_rate = config['model']['attn_drop_rate']
        proj_drop_rata = config['model']['proj_drop_rata']
        
        norm_layer = nn.LayerNorm
        if not isinstance(depth[0], (int, float)):
            depth = [int(d) for d in depth]
        total_depth = int(sum(depth))
        drop_path_rate = float(config['model'].get('drop_path_rate', '0.1'))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.patch_embed = PatchEmbed(
            in_channels, embed_dim[0], kernel_size=3, stride=1, padding=1)

        
        self.stage1_units = nn.ModuleList([
            FreqMambaParallelUnit(
                dim=embed_dim[0], resolution=model_resolution, 
                num_heads=num_heads[0], stripeWidth=split_size[0],
                network_depth=total_depth, currentDepth=i,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                drop_path=dpr[i],
                use_multi_scale=False, enc_mode=True, norm_layer=norm_layer
            ) for i in range(depth[0])
        ])
        self.stage1_refines = nn.ModuleList([
            RefinementBlock(dim=embed_dim[0], kernel_size=3)
            for _ in range(depth[0])
        ])
       
        self.merge1 = DownSample(in_channels=embed_dim[0], out_channels=embed_dim[1],
                                 patch_size=2, kernel_size=5)

        
        self.stage2_units = nn.ModuleList([
            FreqMambaParallelUnit(
                dim=embed_dim[1], resolution=model_resolution//2,
                num_heads=num_heads[1], stripeWidth=split_size[1],
                network_depth=total_depth, currentDepth=np.sum(depth[:1])+i,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                drop_path=dpr[np.sum(depth[:1])+i],
                use_multi_scale=False, enc_mode=True, norm_layer=norm_layer
            ) for i in range(depth[1])
        ])
        
        self.stage2_refines = nn.ModuleList([
            RefinementBlock(dim=embed_dim[1], kernel_size=3)
            for _ in range(depth[1])
        ])
       
        self.merge2 = DownSample(in_channels=embed_dim[1], out_channels=embed_dim[2],
                                 patch_size=2, kernel_size=3)

        
        self.stage3_units = nn.ModuleList([
            FreqMambaParallelUnit(
                dim=embed_dim[2], resolution=model_resolution//4,
                num_heads=num_heads[2], stripeWidth=split_size[2],
                network_depth=total_depth, currentDepth=np.sum(depth[:2])+i,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                drop_path=dpr[np.sum(depth[:2])+i],
                use_multi_scale=False, enc_mode=True, norm_layer=norm_layer
            ) for i in range(depth[2])
        ])
      
        self.stage3_refines = nn.ModuleList([
            RefinementBlock(dim=embed_dim[2], kernel_size=3)
            for _ in range(depth[2])
        ])
       
        self.upMerge1 = UpSample(in_channels=embed_dim[2], out_channels=embed_dim[3],
                                 patch_size=2, kernel_size=None)

      
        self.stage4_units = nn.ModuleList([
            FreqMambaParallelUnit(
                dim=embed_dim[3], resolution=model_resolution//2,
                num_heads=num_heads[3], stripeWidth=split_size[3],
                network_depth=total_depth, currentDepth=np.sum(depth[:3])+i,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                drop_path=dpr[np.sum(depth[:3])+i],
                use_multi_scale=False, enc_mode=False, norm_layer=norm_layer
            ) for i in range(depth[3])
        ])
        
        self.stage4_refines = nn.ModuleList([
            RefinementBlock(dim=embed_dim[3], kernel_size=3)
            for _ in range(depth[3])
        ])
        
        self.upMerge2 = UpSample(in_channels=embed_dim[3], out_channels=embed_dim[4],
                                 patch_size=2, kernel_size=None)

        
        self.stage5_units = nn.ModuleList([
            FreqMambaParallelUnit(
                dim=embed_dim[4], resolution=model_resolution,
                num_heads=num_heads[4], stripeWidth=split_size[4],
                network_depth=total_depth, currentDepth=np.sum(depth[:4])+i,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rata=proj_drop_rata,
                drop_path=dpr[np.sum(depth[:4])+i],
                use_multi_scale=False, enc_mode=False, norm_layer=norm_layer
            ) for i in range(depth[4])
        ])
        
        self.stage5_refines = nn.ModuleList([
            RefinementBlock(dim=embed_dim[4], kernel_size=3)
            for _ in range(depth[4])
        ])

      
        self.conv = nn.Conv2d(embed_dim[4], 3, kernel_size=3, stride=1, padding=1)

        

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, L, 24]
        skip_list = []

        
        for idx, unit in enumerate(self.stage1_units):
            x = unit(x)  
            x = bchw_2_blc(self.stage1_refines[idx](blc_2_bchw(x)))
        skip_list.append(x) 
        x = bchw_2_blc(self.merge1(blc_2_bchw(x))) 

        for idx, unit in enumerate(self.stage2_units):
            x = unit(x)
            x = bchw_2_blc(self.stage2_refines[idx](blc_2_bchw(x)))
        skip_list.append(x) 
        x = bchw_2_blc(self.merge2(blc_2_bchw(x))) 

       
        for idx, unit in enumerate(self.stage3_units):
            x = unit(x)
            x = bchw_2_blc(self.stage3_refines[idx](blc_2_bchw(x)))
        x = bchw_2_blc(self.upMerge1(blc_2_bchw(x))) 

       
        x = x + skip_list[-1] 
        for idx, unit in enumerate(self.stage4_units):
            x = unit(x)
            x = bchw_2_blc(self.stage4_refines[idx](blc_2_bchw(x)))
        x = bchw_2_blc(self.upMerge2(blc_2_bchw(x))) 
        x = x + skip_list[-2] 
        for idx, unit in enumerate(self.stage5_units):
            x = unit(x)
            x = bchw_2_blc(self.stage5_refines[idx](blc_2_bchw(x)))

       
        return self.conv(blc_2_bchw(x))

    

    def forward(self, x):
        H, W = x.shape[2:]
        
       
        F = self.forward_features(x) # [B, 3, H, W]
        
        
        out = x + F 
        
        coarseDehazedImage = out[:, :, :H, :W]
        refinedDehazedImage = coarseDehazedImage  
        
        return coarseDehazedImage, refinedDehazedImage
