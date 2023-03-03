import torch
from torch import nn

class MLPModel(nn.Module):
    def __init__(self, lookback, depth, hidden_dim, pred_samples, activation, act_last):
        super(MLPModel, self).__init__()

        if depth >= 2:
            self.ll_in = nn.Linear(lookback, hidden_dim)
            self.ll_h = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 2)])
            self.ll_out = nn.Linear(hidden_dim, pred_samples)
        else:
            self.ll = nn.Linear(lookback, pred_samples)

        self.depth = depth
        self.activation = activation
        self.act_last = act_last

    def forward(self, x):
        if self.depth >= 2:
            x = self.activation(self.ll_in(x))

            for l in self.ll_h:
                x = self.activation(l(x))

            x = self.ll_out(x)
        else:
            x = self.ll(x)

        if self.act_last:
            x = self.activation(x)

        return x

    def get_json_weights(self):
        out_dict = {}
        out_dict['depth'] = self.depth
        out_dict['act_last'] = self.act_last

        if self.depth >= 2:
            out_dict['lls'] = [
                {
                    'weight': self.ll_in.weight.tolist(),
                    'bias': self.ll_in.bias.tolist(),
                }, 
            ]

            for ll in self.ll_h:
                out_dict['lls'].append({
                    'weight': ll.weight.tolist(),
                    'bias': ll.bias.tolist(),
                })

            out_dict['lls'].append({
                'weight': self.ll_out.weight.tolist(),
                'bias': self.ll_out.bias.tolist(),
            })
        else:
            out_dict['lls'] = [
                {
                    'weight': self.ll.weight.tolist(),
                    'bias': self.ll.bias.tolist(),
                },
            ]

        return out_dict

class RNNWrapper(nn.Module):
    def __init__(self, obs_dim, hidden_dim, update_module, out_samples):
        super(RNNWrapper, self).__init__()
        self.ll_in = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.ll_out = nn.Linear(hidden_dim, obs_dim)
        self.update_module = update_module
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.out_samples = out_samples
        assert out_samples % obs_dim == 0

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.obs_dim)
        
        outs = []

        h = torch.zeros((x.shape[0], self.hidden_dim), dtype=torch.float32)

        for n_out in range(x.shape[1]):
            x_in = x[:, n_out]
            h_in = self.ll_in(x_in)
            h = h + h_in
            h = self.update_module(h)
            outs.append(self.ll_out(h))

        outs = outs[-(self.out_samples // self.obs_dim):]

        return torch.cat(outs, dim=1)

    def get_json_weights(self):
        out_dict = {}
        out_dict['type'] = 'rnn'
        out_dict['hidden_dim'] = self.hidden_dim
        out_dict['obs_dim'] = self.obs_dim
        out_dict['ll_in'] = {
            'weight': self.ll_in.weight.tolist(),
        }
        out_dict['ll_out'] = {
            'weight': self.ll_out.weight.tolist(),
            'bias': self.ll_out.bias.tolist(),
        }
        out_dict['update_module'] = self.update_module.get_json_weights()
        return out_dict
    
class LSTM(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_layers, out_samples):
        super(LSTM, self).__init__()
        self.obs_dim = obs_dim
        self.out_samples = out_samples
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim, num_layers=n_layers, proj_size=obs_dim, batch_first=True)
        assert out_samples % obs_dim == 0

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.obs_dim)
        outs = self.lstm(x)[0]
        outs = outs.reshape(outs.shape[0], -1)
        outs = outs[:, -self.out_samples:]

        return outs

    def get_json_weights(self):
        return {}


class TCN(nn.Module):
    def __init__(self, obs_dim, hidden_dim, depth, activation):
        super(TCN, self).__init__()
        self.conv_in = nn.Conv1d(obs_dim, hidden_dim, 2, dilation=1)
        self.convs_h = nn.ModuleList([nn.Conv1d(hidden_dim, hidden_dim, 2, dilation=2**i) for i in range(1, depth)])
        self.conv_out = nn.Conv1d(hidden_dim, obs_dim, 2, dilation=2**depth)
        self.activation = activation
        self.obs_dim = obs_dim

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.obs_dim).permute(0, 2, 1)
        x = self.activation(self.conv_in(x))

        for c in self.convs_h:
            x = self.activation(c(x))

        x = self.conv_out(x).permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)

        return x

    def get_json_weights(self):
        return {} # TODO

class TCN_2(nn.Module):
    def __init__(self,):
        super(TCN_2, self).__init__()

        # takes 4 / 8 / 16 best in one go and applies some MLP on top

class NAMPModel(nn.Module):
    def __init__(self, module):
        super(NAMPModel, self).__init__()
        self.module = module

    def forward(self, x):
        x = self.module(x)
        return x

    def get_json_weights(self):
        return self.module.get_json_weights()
