import torch
import torch.nn as nn
import math

import torch.nn.functional as F

import time
import numpy as np

from utils.options import args

device = torch.device(f"cuda:{args.gpus[0]}")


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit, stage):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)
  
  def normalize(self, x, range_ = 2):
    new_s = range_ / 4
    m = torch.mean(x)
    s = torch.std(x)
    return (x - m) / s * new_s
      
  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      x = self.normalize(x)
      weight_q = self.uniform_q(x)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit, stage):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)
  
  def normalize(self, x, range_ = 2):
    new_s = range_ / 4
    m = torch.mean(x)
    s = torch.std(x)
    return (x - m) / s * new_s

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      #x = self.normalize(x)
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


def conv2d_Q_fn(w_bit, stage):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit, stage = stage)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q
