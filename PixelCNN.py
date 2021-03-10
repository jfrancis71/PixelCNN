#Simplified implementation of PixelCNN
#@misc{oord2016pixel,
#      title={Pixel Recurrent Neural Networks}, 
#      author={Aaron van den Oord and Nal Kalchbrenner and Koray Kavukcuoglu},
#      year={2016},
#      eprint={1601.06759},
#      archivePrefix={arXiv},
#      primaryClass={cs.CV}
#}
#https://arxiv.org/abs/1601.06759


import torch
import torch.nn as nn

class PixelCNNMaskConv2d(nn.Conv2d):

    def __init__(self, mask_type, kernel_size, num_input_channels, in_data_channel_width, out_data_channel_width):
        in_channels = num_input_channels*in_data_channel_width
        out_channels = num_input_channels*out_data_channel_width
        middle = kernel_size//2
        super(PixelCNNMaskConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=middle)
        self.mask = torch.nn.Parameter(torch.zeros((out_channels,in_channels,kernel_size,kernel_size)), requires_grad=False)
        self.mask[:,:,:middle,:] = 1.0
        self.mask[:,:,middle,:middle] = 1.0
        if mask_type == "A":
            for c in range(1,num_input_channels):
                #If first layer then the data channel width is 1
                self.mask[c*out_data_channel_width:(c+1)*out_data_channel_width,:c,middle,middle] = 1.0
        else:
            for c in range(num_input_channels):
                self.mask[c*out_data_channel_width:(c+1)*out_data_channel_width,:(c+1)*in_data_channel_width,middle,middle] = 1.0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_input_channels, in_data_channel_width, num_h=None):
        super(ResidualBlock, self).__init__()
        self.layer1 = PixelCNNMaskConv2d("B", kernel_size=1, num_input_channels=num_input_channels, in_data_channel_width=in_data_channel_width, out_data_channel_width=in_data_channel_width//2)
        self.layer2 = PixelCNNMaskConv2d("B", kernel_size=3, num_input_channels=num_input_channels, in_data_channel_width=in_data_channel_width//2, out_data_channel_width=in_data_channel_width//2)
        self.layer3 = PixelCNNMaskConv2d("B", kernel_size=1, num_input_channels=num_input_channels, in_data_channel_width=in_data_channel_width//2, out_data_channel_width=in_data_channel_width)
        if num_h is not None:
            self.conditional_prj = nn.Linear(num_h, num_input_channels*in_data_channel_width//2, bias=False)
        else:
            self.conditional_prj = None
    
    def forward(self, x, h=None):
        r = self.layer1(x)
        r = nn.ReLU()(r)
        r = self.layer2(r)
        if h is not None:
            prj = self.conditional_prj(h)
            add_r = r.permute((2,3,0,1)) + prj
            r = add_r.permute((2,3,0,1))

        r = nn.ReLU()(r)
        r = self.layer3(r)
        r = nn.ReLU()(r)
        return x+r

class PixelCNN(nn.Module):
    def __init__(self, num_input_channels, num_distribution_params, num_h=None, num_blocks=15, data_channel_width=256):
        super(PixelCNN, self).__init__()
        self.input_layer = PixelCNNMaskConv2d("A", kernel_size=7, num_input_channels=num_input_channels, in_data_channel_width=1, out_data_channel_width=data_channel_width)
        blocks = [ ResidualBlock(num_input_channels, data_channel_width, num_h) for _ in range(num_blocks) ]
        self.blocks = nn.ModuleList(blocks)
        self.layer1 = PixelCNNMaskConv2d("B", kernel_size=1, num_input_channels=num_input_channels, in_data_channel_width=data_channel_width, out_data_channel_width=data_channel_width//2)
        self.layer2 = PixelCNNMaskConv2d("B", kernel_size=1, num_input_channels=num_input_channels, in_data_channel_width=data_channel_width//2, out_data_channel_width=num_distribution_params)
    
    def forward(self, x, h=None):
        x = self.input_layer(x)
        for r in self.blocks:
            x = r(x, h)
        x = nn.ReLU()(x)
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        return x
