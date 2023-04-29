import torch
import torch.nn as nn
input_tensor = torch.randn(1, 3, 12, 12) # batch_size x channels x height x width
conv_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
output_tensor = conv_layer.cuda()(input_tensor.cuda())
