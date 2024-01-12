import torch
import torch.nn as nn


def upsample_block(x, filters, size, stride=2):
    """
  x - the input of the upsample block
  filters - the number of filters to be applied
  size - the size of the filters
  """

    # TODO your code here
    # transposed convolution
    x = nn.ConvTranspose2d(in_channels=x.size(1),
                           out_channels=filters,
                           kernel_size=size,
                           stride=stride)(x)
    # BN
    x = nn.BatchNorm2d(filters)(x)
    # relu activation
    x = nn.ReLU()(x)
    return x


in_layer = torch.rand((32, 32, 128, 128))

filter_sz = 4
num_filters = 16

for stride in [2, 4, 8]:
    x = upsample_block(in_layer, num_filters, filter_sz, stride)
    print('in shape: ', in_layer.shape, ' upsample with filter size ', filter_sz, '; stride ', stride, ' -> out shape ',
          x.shape)
