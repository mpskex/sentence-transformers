import os
import json
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable, grad


class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        # only getting the sign of input, binarizing the embedding
        result = input.sign()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator bypass the gradient
        return grad_output


class HashLayer(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 momentum: float=0.9,
                 bias: bool=False,
                 ):
        """Embedding Hash Layer

        Args:
            embedding_dim (int): 
            momentum (float, optional): momentum of moving mean. Defaults to 0.9.
        """
        super().__init__()
        
        self.config_keys = ["in_features", "out_features", "bias", "momentum"]
        
        self.transform = torch.nn.Linear(in_features, out_features, bias=bias)
        self.momentum = momentum
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_buffer("_mu", torch.zeros((1, self.out_features)))

    def forward(self, features: Dict[str, torch.Tensor]):
        input = self.transform(features['sentence_embedding'])
        if self.training:
            mean = input.mean([0]).view(1, -1)
            with torch.no_grad():
                self._mu.copy_(
                    (self.momentum * self._mu) + (1.0 - self.momentum) * mean
                )
        else:
            mean = self._mu
        # get output
        h = input - mean
        y = StraightThroughEstimator.apply(h)
        features.update({'binary_embedding_approx': h,
                         'binary_embedding': y})
        return features
    
    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        model = HashLayer(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
