import torch
import torch.nn as nn
import torchvision.models as models

class DepthwiseSeparableConv(nn.Module):
    """
    MobileNet-inspired Depthwise Separable Convolution to reduce model size
    and inference time without losing accuracy.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, stride=stride, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileResnetBlock(nn.Module):
    """
    Residual block utilizing depthwise separable convolutions,
    instance normalization, and a dropout layer with probability 0.
    """
    def __init__(self, channels):
        super(MobileResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0), # Probability of zero as specified in paper
            nn.ReflectionPad2d(1),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    Generator network: Transformed ResNet architecture generating
    sharp images $I_S$ from blurry images $I_B$.
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(Generator, self).__init__()

        # Initial Convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling: Two strided convolution blocks
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # 9 MobileResnet Blocks
        mult = 2 ** n_downsampling
        for i in range(9):
            model += [MobileResnetBlock(ngf * mult)]

        # Upsampling: Two transposed convolution blocks
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]

        # Final output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # Global skip connection (ResOut): $I_S = I_B + I_R$
        residual = self.model(x)
        out = x + residual
        # Clamp output to match the Tanh output domain [-1, 1]
        return torch.clamp(out, min=-1.0, max=1.0)

class Discriminator(nn.Module):
    """
    Critic network (PatchGAN discriminator).
    Evaluates concatenated input of (Blurry + Sharp/Generated).
    """
    def __init__(self, in_channels=6, ndf=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: No InstanceNorm
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 (Stride 1 as per standard PatchGAN)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Layer: No InstanceNorm, No LeakyReLU
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    """
    Calculates MSE over VGG16 feature maps to retain structural knowledge.
    Uses the relu3_3 layer outputs.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        # Extract layers up to relu3_3 (index 15) -> slice up to 16
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()

        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return self.mse_loss(gen_features, target_features)
