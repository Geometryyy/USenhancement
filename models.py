import torch
import torch.nn as nn
import torch.nn.functional as F
        

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        g_x = F.max_pool2d(self.g(x), (8, 8)).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = F.max_pool2d(self.phi(x), (8, 8)).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = f / f.shape[-1] 
        y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, h, w)
        z = self.W(y) + x
        return z


class LocalAttention(nn.Module):
    def __init__(self, k, in_channels):
        super(LocalAttention, self).__init__()
        self.softmax = nn.Softmax2d()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2*k+1, padding=k, groups=in_channels, bias=False)

    def forward(self, x):
        x = self.conv(self.softmax(x) * x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.tanh(avg_out + max_out)
        return scale * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.tanh(self.conv1(scale))
        return scale * x


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7, channel_attn=True):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio) if channel_attn else False
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        if self.channel_attention:
            x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CONVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, upSampledsize=None):
        super(CONVBlock, self).__init__()
        self.block = []
        if upSampledsize:
            self.block.append(nn.Upsample(size=upSampledsize, mode='bicubic'))
        self.block += [nn.Conv2d(in_channels, out_channels, kernel),
                       nn.InstanceNorm2d(out_channels),
                       nn.LeakyReLU(0.2, inplace=True), 
                       CBAM(out_channels)
                       ]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, mid_channels, 3),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(mid_channels, out_channels, 3),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upSample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x, x2):
        x = self.conv(torch.cat([x2, self.upSample(x)], dim=1))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(planes, planes, kernel_size=3, bias=False)
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, kernel_size=7, bias=False)
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) # 256
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) #512, 32
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),  # relu3-2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),  # relu4-2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),  # relu5-2
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.res1 = DoubleConv(512, 512)
        self.res2 = DoubleConv(512, 512)
        self.res3 = DoubleConv(512, 512)
        self.res4 = DoubleConv(512, 512)
        self.res5 = DoubleConv(512, 512)
        
        self.conv1e3d = DoubleConv(64, 64)
        self.cbam1e3d = CBAM(64)
        self.conv2e3d = DoubleConv(128, 64)
        self.cbam2e3d = CBAM(64)
        self.conv3e3d = DoubleConv(256, 64)
        self.cbam3e3d = CBAM(64)
        self.conv4d3d = DoubleConv(512, 64)
        self.cbam4d3d = CBAM(64)
        self.fconv3d = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 256, 3))
        
        self.conv1e2d = DoubleConv(64, 64)
        self.cbam1e2d = CBAM(64)
        self.conv2e2d = DoubleConv(128, 64)
        self.cbam2e2d = CBAM(64)
        self.conv4d2d = DoubleConv(512, 64)
        self.cbam4d2d = CBAM(64)
        self.conv3d2d = DoubleConv(256, 64)
        self.cbam3d2d = CBAM(64)
        self.fconv2d = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 128, 3))
        
        self.conv1e1d = DoubleConv(64, 64)
        self.cbam1e1d = CBAM(64)
        self.conv4d1d = DoubleConv(512, 64)
        self.cbam4d1d = CBAM(64)
        self.conv3d1d = DoubleConv(256, 64)
        self.cbam3d1d = CBAM(64)
        self.conv2d1d = DoubleConv(128, 64)
        self.cbam2d1d = CBAM(64)
        self.fconv1d = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 64, 3))

        self.down4x = nn.MaxPool2d(4)
        self.down2x = nn.MaxPool2d(2)
        self.up8x = nn.Upsample(scale_factor=8, mode='bicubic')
        self.up4x = nn.Upsample(scale_factor=4, mode='bicubic')
        self.up2x = nn.Upsample(scale_factor=2, mode='bicubic')
        
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, e1, e2, e3, e4):
        x = self.res1(e4) + e4
        x = self.res2(x) + x
        x = self.res3(x) + x
        x = self.res4(x) + x
        d4 = self.res5(x) + x
        d3 = torch.cat([self.cbam1e3d(self.conv1e3d(self.down4x(e1))),
                        self.cbam2e3d(self.conv2e3d(self.down2x(e2))),
                        self.cbam3e3d(self.conv3e3d(e3)),
                        self.cbam4d3d(self.conv4d3d(self.up2x(d4)))], dim=1)
        d3 = self.fconv3d(d3)
        d2 = torch.cat([self.cbam1e2d(self.conv1e2d(self.down2x(e1))),
                        self.cbam2e2d(self.conv2e2d(e2)),
                        self.cbam3d2d(self.conv3d2d(self.up2x(d3))),
                        self.cbam4d2d(self.conv4d2d(self.up4x(d4)))], dim=1)
        d2 = self.fconv2d(d2)
        d1 = torch.cat([self.cbam1e1d(self.conv1e1d(e1)),
                        self.cbam2d1d(self.conv2d1d(self.up2x(d2))),
                        self.cbam3d1d(self.conv3d1d(self.up4x(d3))),
                        self.cbam4d1d(self.conv4d1d(self.up8x(d4)))], dim=1)
        d1 = self.fconv1d(d1)
        x = self.out(d1)
        return x


def adaIN(x, y): # x:content y:style
    x_mean = x.mean(dim=(2, 3), keepdim=True)
    x_std = x.std(dim=(2, 3), keepdim=True) + 1e-6
    y_mean = y.mean(dim=(2, 3), keepdim=True)
    y_std = y.std(dim=(2, 3), keepdim=True) + 1e-6
    out = y_std * (x - x_mean) / x_std + y_mean
    return out
    

class ResEncoder(nn.Module):
    def __init__(self, path='/data/chenliuji/GAN/code/USenhancement/AdaIN/US/resnet/models/classifier_35.pth'):
        super().__init__()
        self.resnet = torch.load(path).model.module
        self.slice1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.layer1
            )
        self.slice2 = self.resnet.layer2
        self.slice3 = self.resnet.layer3
        self.slice4 = self.resnet.layer4
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class AdaINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResEncoder()
        self.decoder = Decoder(1)

    def generate(self, content_images, style_images, alpha=1.0):
        x1, x2, x3, x4 = self.encoder(content_images, output_last_feature=False)
        y1, y2, y3, y4 = self.encoder(style_images, output_last_feature=False)
        e1, e2, e3, e4 = adaIN(x1, y1), adaIN(x2, y2), adaIN(x3, y3), adaIN(x4, y4)
        out = self.decoder(e1, e2, e3, e4)
        return out

    @staticmethod
    def calc_content_loss(o1, o2, o3, o4, e1, e2, e3, e4):
        return F.mse_loss(o1, e1) + F.mse_loss(o2, e2) + F.mse_loss(o3, e3) + F.mse_loss(o4, e4)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean = c.mean(dim=(2, 3), keepdim=True)
            c_std = c.std(dim=(2, 3), keepdim=True) + 1e-6
            s_mean = s.mean(dim=(2, 3), keepdim=True)
            s_std = s.std(dim=(2, 3), keepdim=True) + 1e-6
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        x1, x2, x3, x4 = self.encoder(content_images, output_last_feature=False)
        y1, y2, y3, y4 = self.encoder(style_images, output_last_feature=False)
        e1, e2, e3, e4 = adaIN(x1, y1), adaIN(x2, y2), adaIN(x3, y3), adaIN(x4, y4)
        out = self.decoder(e1, e2, e3, e4)

        o1, o2, o3, o4 = self.encoder(out, output_last_feature=False)
        style_middle_features = self.encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(o1, o2, o3, o4, e1, e2, e3, e4)
        loss_s = self.calc_style_loss((o1, o2, o3, o4), style_middle_features)
        loss = loss_c + lam * loss_s
        return loss, out


class Model(nn.Module):
    def __init__(self, model, device=None):
        super(Model, self).__init__()
        self.model = torch.nn.DataParallel(model, device_ids=device)
        # self.model = model
        
    def forward(self, *args):
        return self.model(*args)
    

