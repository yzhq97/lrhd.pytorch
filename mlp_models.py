import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn
import math
import torch.nn.init as init

from embedder import Embedder


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

        # self.load_weights()
        self.init_weights()

    def load_weights(self):
        checkpoint_path = './logs/plane1/occ_10155655850468db78d106ce0a280f87_jitter_0_model_final.pth'
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint)

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

        # for layer in (self.layers):
        #     init.kaiming_normal_(layer.weight, nonlinearity='relu')
        #     if layer.bias is not None:
        #         init.zeros_(layer.bias)

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
# def exists(val):
#     return val is not None

# def cast_tuple(val, repeat = 1):
#     return val if isinstance(val, tuple) else ((val,) * repeat)
    
# class Sine(nn.Module):
#     def __init__(self, w0 = 1.):
#         super().__init__()
#         self.w0 = w0
#     def forward(self, x):
#         return torch.sin(self.w0 * x)

# # siren layer

# class Siren(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         w0 = 1.,
#         c = 6.,
#         is_first = False,
#         use_bias = True,
#         activation = None,
#         dropout = 0.
#     ):
#         super().__init__()
#         self.dim_in = dim_in
#         self.is_first = is_first

#         weight = torch.zeros(dim_out, dim_in).float()
#         bias = torch.zeros(dim_out).float() if use_bias else None
#         self.init_(weight, bias, c = c, w0 = w0)

#         self.weight = nn.Parameter(weight)
#         self.bias = nn.Parameter(bias) if use_bias else None
#         self.activation = Sine(w0) if activation is None else activation
#         self.dropout = nn.Dropout(dropout)

#     def init_(self, weight, bias, c, w0):
#         dim = self.dim_in

#         w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
#         weight.uniform_(-w_std, w_std)

#         if exists(bias):
#             bias.uniform_(-w_std, w_std)

#     def forward(self, x):
#         out =  F.linear(x, self.weight, self.bias)
#         out = self.activation(out)
#         out = self.dropout(out)
#         return out

# # siren network

        
# class MLP3D(nn.Module):
#     def __init__(
#         self,
#         hidden_neurons,
#         out_size,
#         w0 = 1.,
#         w0_initial = 30.,
#         use_bias = True,
#         final_activation = None,
#         dropout = 0.,
#         use_leaky_relu=False,
#         multires=10,
#         output_type=None,
#         move=False,
#         **kwargs,
#     ):
#         super().__init__()
#         dim_in = 3
        
#         self.num_layers = len(hidden_neurons)

#         self.layers = nn.ModuleList([])

        
#         for ind, _ in enumerate(hidden_neurons):
#             is_first = ind == 0
#             layer_w0 = w0_initial if is_first else w0
#             layer_dim_in = dim_in if is_first else hidden_neurons[ind-1]

#             layer = Siren(
#                 dim_in = layer_dim_in,
#                 dim_out = hidden_neurons[ind],
#                 w0 = layer_w0,
#                 use_bias = use_bias,
#                 is_first = is_first,
#                 dropout = dropout
#             )

#             self.layers.append(layer)

#         final_activation = nn.Identity() if not exists(final_activation) else final_activation
#         self.last_layer = Siren(dim_in = hidden_neurons[-1], dim_out = out_size, w0 = w0, use_bias = use_bias, activation = final_activation)

#     def forward(self, x, mods = None):
#         coords_org = x["coords"].clone().detach()
#         x = coords_org
        
#         mods = cast_tuple(mods, self.num_layers)

#         for layer, mod in zip(self.layers, mods):
#             x = layer(x)

#             if exists(mod):
#                 x *= rearrange(mod, 'd -> () d')
            
        
#         x = self.last_layer(x)
#         # print('coords_org max', coords_org.max())
#         # print('coords_org min', coords_org.min())
        
#         # print('x max', x.max())
#         # print('x min', x.min())
#         return {"model_in": coords_org, "model_out": x}
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