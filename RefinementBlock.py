import torch
import torch.nn as nn
import torch.nn.init as init

# ============== 注意力模块 ==============

class PALayer(nn.Module):
   
    def __init__(self, channel, reduction=8):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()  

    def _init_weights(self):  
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1)  # [B, 1, C]
        y = self.conv(y)                    # [B, 1, C]
        y = self.sigmoid(y)                 # [B, 1, C]
        y = y.permute(0, 2, 1).unsqueeze(-1) # [B, C, 1, 1]
        return x * y.expand_as(x)


# ============== RefinementBlock ==============

class RefinementBlock(nn.Module):
    '''
    FFN模块: LN → DwConv→GELU→ECA→PA → Skip(s')
    接口: __init__(dim, kernel_size=3)  
    '''
    def __init__(self, dim, kernel_size=3):  
        super(RefinementBlock, self).__init__()
        
        # 1. LayerNorm
        self.ln = nn.LayerNorm(dim)
        
        # 2. DwConv 3×3 (Depth-wise Conv)
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2, 
            groups=dim, 
            bias=False
        )
        
        # 3. 激活函数
        self.act = nn.GELU()
    
        self.ca = ECAAttention(kernel_size=3)
        self.pa = PALayer(channel=dim) 
        
        self.skip_scale = nn.Parameter(torch.ones(dim))

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        residual = x  
        
        #  LayerNorm
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        
        # DwConv + GELU
        x = self.dwconv(x)
        x = self.act(x)
        
        x = self.ca(x)
        x = self.pa(x) 
       
        out = x + self.skip_scale.view(1, -1, 1, 1) * residual
        
        return out
