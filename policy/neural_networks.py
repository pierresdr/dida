import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch._jit_internal import _copy_to_script_wrapper
from typing import Iterator, Dict

def get_number_of_parameters(network):
    if hasattr(network,'n_params'):
        return network.n_params
    else: 
        nb_param = 0
        for parameter in network.parameters():
            nb_param += parameter.numel()
        return nb_param

class mlp(nn.Module):
    def __init__(self, n_neurons, activation=nn.ReLU(), output_activation=nn.Identity(),
                all_actions=False,):
        super(mlp, self).__init__()
        layers = []
        for i,j in zip(n_neurons[:-1],n_neurons[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers[-1] = output_activation
        self.linear =  nn.Sequential(*layers)

    def forward(self, state, actions):
        return self.linear(torch.cat((state,actions.reshape(state.size(0),-1)),dim=1))

class mlp_max(nn.Module):
    def __init__(self, n_neurons, activation=nn.ReLU(), output_activation=nn.Identity(),
                all_actions=False,):
        super(mlp_max, self).__init__()
        layers = []
        for i,j in zip(n_neurons[:-1],n_neurons[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers[-1] = output_activation
        self.linear =  nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        probs = self.linear(state)
        return self.softmax(probs)

    def sample(self, state):
        probs = torch.cumsum(self(state), dim=1)
        return torch.sum(torch.rand(probs.shape[0]).reshape(-1,1)>probs, dim=1)


class cnn(nn.Module):
    @staticmethod
    def output_size(l_in, n_channels, kernel_size):
        return l_in - (kernel_size-1)*len(n_channels)


    def __init__(self, n_channels, n_neurons, kernel_size, activation=nn.ReLU(), output_activation=nn.Identity()):
        super(cnn, self).__init__()
        layers = []
        for i,j in zip(n_channels[:-1],n_channels[1:]):
            layers.append(nn.Conv1d(i,j,kernel_size=kernel_size))
            layers.append(activation)
        self.conv = nn.Sequential(*layers)

        layers = []
        for i,j in zip(n_neurons[:-1],n_neurons[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers[-1] = output_activation
        self.linear = nn.Sequential(*layers)

    def forward(self, state, actions):
        encoded_actions = self.conv(actions.transpose(1,2))
        return self.linear(torch.cat((state,encoded_actions.reshape(state.size(0),-1)),dim=1))

         
class AutoregLayer(nn.Module):
    def __init__(self, state_dim, in_action_dim, out_action_dim, delay,):
        super(AutoregLayer, self).__init__()
        self.in_action_dim = in_action_dim
        self.out_action_dim = out_action_dim
        self.delay = delay
        self.linear = nn.Linear(in_action_dim*delay, out_action_dim*delay)
        mask = torch.stack([torch.arange(in_action_dim*delay)<(1+l//out_action_dim)*in_action_dim for l in range(out_action_dim*delay)])
        self.register_buffer('mask', mask)
        self.cond_linear = nn.Linear(state_dim, out_action_dim)

    def forward(self, state, actions,):
        trunc = actions.size(1)//self.in_action_dim
        output = F.linear(actions, 
                        self.linear.weight[:trunc*self.out_action_dim,:trunc*self.in_action_dim] 
                        * self.mask[:trunc*self.out_action_dim,:trunc*self.in_action_dim], 
                        self.linear.bias[:trunc*self.out_action_dim])
        return  output+self.cond_linear(state).tile(trunc)
        

class AutoregNet(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, state_dim, action_dims, delay, activation=nn.ReLU(), output_activation=nn.Identity()):
        super(AutoregNet, self).__init__()
        self.delay = delay
        self.action_dim = action_dims[-1]
        for k, (i,j) in enumerate(zip(action_dims[:-1],action_dims[1:])):
            self.add_module('AutoregLayer{}'.format(k+1), AutoregLayer(state_dim=state_dim, in_action_dim=i, out_action_dim=j, delay=delay))
            if k==len(action_dims)-2:
                self.add_module(output_activation.__class__.__name__, output_activation)
            else:
                self.add_module(activation.__class__.__name__+str(k), copy.deepcopy(activation))
        self.n_params = get_number_of_parameters(self)
        for module in self:
            if isinstance(module, AutoregLayer):
                self.n_params -= torch.sum(~module.mask).item()

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def forward(self, state, actions,):
        actions = actions.reshape(state.size(0),-1)
        for module in self:
            if isinstance(module, AutoregLayer):
                actions = module(state, actions)
            else:
                actions = module(actions)
        return actions.reshape(state.size(0),-1,self.action_dim)