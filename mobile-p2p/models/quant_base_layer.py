import torch
import torch.nn as nn
import platform
python_v = str(platform.python_version()).split('.')[0]



class Slice(nn.Module):
    def __init__(self, output_num=2, dim=1):
        if python_v == '2':
            super(Slice, self).__init__()
        elif python_v == '3':
            super().__init__()
        else:
            assert False
        self.output_num = output_num
        self.dim = dim

    def forward(self, x):
        return torch.chunk(x, self.output_num, self.dim)



class Concat(nn.Module):
    def __init__(self, dim=1):
        if python_v == '2':
            super(Concat, self).__init__()
        elif python_v == '3':
            super().__init__()
        else:
            assert False
        self.dim = dim

    def forward(self, *inputs):
        if(len(inputs) != 1):
            return torch.cat(inputs, dim=self.dim)
        elif(len(inputs[0]) != 1):
            return torch.cat(inputs[0], dim=self.dim)
        else:
            assert False



class Eltwise(nn.Module):
    def __init__(self):
        if python_v == '2':
            super(Eltwise, self).__init__()
        elif python_v == '3':
            super().__init__()
        else:
            assert False

    def forward(self, *inputs):
        if(len(inputs) == 2):
            return inputs[0] + inputs[1]
        elif(len(inputs[0])==2):
            return inputs[0][0] + inputs[0][1]
        else:
            assert False



class ShuffleChannel(nn.Module):
    def __init__(self, groups=2, mini_group=4, size=None):
        if python_v == '2':
            super(ShuffleChannel, self).__init__()
        elif python_v == '3':
            super().__init__()
        else:
            assert False
        self.groups = groups
        self.mini_group = mini_group
        self.size = size

    def forward(self, input):
        batchsize, num_channels, height, width = input.data.size()
        channels_per_group = num_channels // self.groups // self.mini_group
        # reshape
        input = input.view(batchsize, self.groups,
        channels_per_group, self.mini_group, height, width)

        input = torch.transpose(input, 1, 2).contiguous()

        # flatten
        out = input.view(batchsize, -1, height, width)

        return out

