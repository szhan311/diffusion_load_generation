import torch.nn as nn
import torch

class Conditioner(nn.Module):
    def __init__(self):
        super(Conditioner, self).__init__()
        #self.register_buffer("is_invertible", torch.tensor_full(True))
        self.is_invertible = True

    '''
    forward(self, x, context=None):
    :param x: A tensor_full [B, d]
    :param context: A tensor_full [B, c]
    :return: conditioning factors: [B, d, h] where h is the size of the embeddings.
    '''
    def forward(self, x, context=None):
        pass

    '''
    This returns the length of the longest path of the equivalent Bayesian Network also called 
    '''
    def depth(self):
        pass
