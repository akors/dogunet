import functools
from typing import Dict, List

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


class ActivationsLogger():
    def __init__(self, model: torch.nn.Module, writer: SummaryWriter, layers: List[str]):
        self.__model = model
        self.__writer = writer
        self.__layers = layers.copy()
        self.__handles = []

        self.activations: Dict[str, List[torch.Tensor]] = {l: [] for l in layers}

    def _forward_hook(self, name, mod, inp, outp):
        activations: torch.Tensor = outp.cpu().detach()
        self.activations[name].append(activations)        


    def _register_hook(self, name):
        m = self.__model.get_submodule(name)
        h = m.register_forward_hook(functools.partial(self._forward_hook, name))
        self.__handles.append(h)

    def enable(self):
        # skip if already enabled
        if len(self.__handles) > 0:
            return

        for layer in self.__layers:
            self._register_hook(layer)
    
    def disable(self):
        for h in self.__handles:
            h.remove()
        
        self.__handles.clear()

    def flush(self, global_step: int, phase: str="train"):
        for layer, act in self.activations.items():
            # concat all tensor in list along the batch dimension, assumed to be dim=0
            aggregate = torch.concat(act, dim=0)

            self.__writer.add_histogram(tag=f"Activations/{layer}/{phase}", values=aggregate, global_step=global_step)
            act.clear()
        
    def clear(self):
        for l in self.activations.values():
            l.clear()
        