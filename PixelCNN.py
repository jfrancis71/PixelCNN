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


class PixelCNN(nn.Module):

    def __init__(self, num_input_channels, num_distribution_params, kernel_size=3, num_layers=5, num_hidden_features=64, num_spatial_conditional_channels=None):
        super(PixelCNN, self).__init__()
        h = num_hidden_features
        if num_spatial_conditional_channels is not None:
            self.layer_spatial_prj = nn.Conv2d(num_spatial_conditional_channels, h*num_input_channels, kernel_size=(1,1))
        else:
            self.layer_spatial_prj = None
        self.input_layer = PixelCNNMaskConv2d("A", kernel_size, num_input_channels, 1, h)
        self.layers = nn.ModuleList([PixelCNNMaskConv2d("B", kernel_size, num_input_channels, h, h) for _ in range(num_layers)])
        self.output_layer = PixelCNNMaskConv2d("B", kernel_size, num_input_channels, h, num_distribution_params)

    def forward(self, x, spatial_conditional=None):
        x = self.input_layer(x)
        if spatial_conditional is not None:
            prj = self.layer_spatial_prj(spatial_conditional)
            x += prj
        x = nn.ReLU()(x)
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.output_layer(x)
        return x
