import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn
import math
import torch.nn.init as init

from embedder import Embedder
from typing import Optional, List


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        lora_rank = 16
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(Linear(in_size, hidden_neurons[0], bias=use_bias, r=lora_rank))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias, r=lora_rank)
            )
        self.layers.append(Linear(hidden_neurons[-1], out_size, bias=use_bias, r=lora_rank))

        self.init_weights()
        # self.load_weights()

    def load_weights(self):
        checkpoint_path = './logs/plane1/occ_10155655850468db78d106ce0a280f87_jitter_0_model_final.pth'
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint, strict=False)

    def init_(self, weight, bias, dim_in, is_first, c=6, w0=1):
        w_std = (1 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        with torch.no_grad():
            weight.uniform_(-w_std, w_std)
            if bias is not None:
                bias.uniform_(-w_std, w_std)
    
    def init_weights(self):
        c = 6
        for i, layer in enumerate(self.layers):
            dim_in = layer.in_features 
            is_first = (i == 0)  
            if is_first:
                w0 = 30
            else:
                w0 = 1
            self.init_(layer.weight, layer.bias, dim_in, is_first, c, w0)

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)

        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        
        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"


        # x = dist.Bernoulli(logits=x).logits


        return {"model_in": coords_org, "model_out": x}





# ---------------------------------------------------
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.bias.requires_grad = False  # TODO
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)  # TODO
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
# ---------------------------------------------------

class MLP2D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=2,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)


        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        elif self.output_type == "pixel":
            x = torch.tanh(x)
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"

        # x = dist.Bernoulli(logits=x).logits

        return {"model_in": coords_org, "model_out": x}