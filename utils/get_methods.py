import torch
from collections import OrderedDict


def get_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_clean_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_out_var(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def get_clean_out_var(out_holder):
    out = torch.vstack(out_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def take_mean(input_ten):
    input_ten = torch.mean(input_ten, dim=0)
    return input_ten


def get_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


