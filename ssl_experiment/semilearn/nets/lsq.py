import torch
import torch.nn as nn
import torch.nn.functional as F
        
def grad_scale(x, scale):
    y_grad = x * scale
    return (x - y_grad).detach() + y_grad

def round_pass(x):
    return x + x.round().detach() - x.detach()

class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=True):
        super(LSQActivationQuantizer, self).__init__()
        assert a_bits <= 8 or a_bits == 32
        self.a_bits = a_bits
        self.alpha = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.init = 0
        if all_positive:
            self.QN = 0
            self.QP = 2 ** a_bits - 1
        else:
            self.QN = - 2 ** (a_bits - 1)
            self.QP = 2 ** (a_bits - 1) - 1

    def init_from(self, x):
        self.alpha.data.copy_(x.abs().mean() * 2 / (self.QP ** 0.5))

    def forward(self, x):
        if self.init == 0 and self.training:
            self.init_from(x)
        self.init += 1

        s_scale = grad_scale(self.alpha, 1.0 / ((self.QP * x.numel()) ** 0.5))
        x = x / s_scale
        x = torch.clamp(x, self.QN, self.QP)
        x = round_pass(x)
        x = x * s_scale
        return x

class LSQQuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=4,
                 w_bits=4):
        super(LSQQuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = LSQActivationQuantizer(a_bits=w_bits, all_positive=False)

    def forward(self, x):
        x = self.activation_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)

        output = F.conv2d(x, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output
    

class LSQQuantLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, a_bits=4, w_bits=4):
        super().__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = LSQActivationQuantizer(a_bits=w_bits, all_positive=False)

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_act = self.activation_quantizer(x)
        return nn.functional.linear(quantized_act, quantized_weight, self.bias)
