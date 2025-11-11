import torch
from torch import nn, backends
backends.cudnn.benchmark = True


class MaskedConv2d(nn.Conv2d):
    ''' This code is from based on: https://github.com/jzbontar/pixelcnn-pytorch '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    
class MaskedConv1d(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}

        # weight shape: [out_channels, in_channels, kernel_size]
        self.register_buffer('mask', torch.ones_like(self.weight))

        k = self.kernel_size[0]
        center = k // 2

        # Mask: block future positions
        self.mask[:, :, center + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN1D(nn.Module):
    def __init__(self, k_dim, z_dim, fm=64, kernel_size=5):
        super().__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.padding = (kernel_size-1) // 2
        #self.embedding = nn.Embedding(k_dim, z_dim)

        self.encode = nn.Sequential(
            MaskedConv1d('A', z_dim, fm, kernel_size, padding=self.padding, bias=False),
            nn.ReLU(inplace=True),

            MaskedConv1d('B', fm, fm, kernel_size, padding=self.padding, bias=False),
            nn.ReLU(inplace=True),
            MaskedConv1d('B', fm, fm, kernel_size, padding=self.padding, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv1d(fm, k_dim, 1)  # output logits per token
        )

    def forward(self, X):
        # X of shape [B, z_dim]
        logits = self.encode(X)  # -> [B, k_dim, L]
        return logits



class PIXELCNN2D(nn.Module):
    ''' This code is from based on: https://github.com/jzbontar/pixelcnn-pytorch '''
    # 1 chanel PixelCNN
    def __init__(self, k_dim, z_dim, kernel_size=3, fm=32):
        super(PIXELCNN2D, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.fm = fm
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1)//2
        self.encode = nn.Sequential(
            MaskedConv2d('A', self.z_dim,  self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            MaskedConv2d('B', self.fm, self.fm, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(self.fm), nn.ReLU(True),
            nn.Conv2d(self.fm, self.k_dim, 1))

    def forward(self, z):
        return self.encode(z)

