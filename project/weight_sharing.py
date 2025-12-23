import torch
import torch.nn as nn
import types

def apply_weight_sharing_resnet(model, R=2):
    master_idx = R - 1
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

    for layer_name in layer_names:
        if not hasattr(model.model, layer_name): continue

        layer_container = getattr(model.model, layer_name)
        num_blocks = len(layer_container)
        if num_blocks <= R:
            continue

        master_block = layer_container[master_idx]
        for i in range(master_idx + 1, num_blocks):
            block = layer_container[i]
            for name, m_child in master_block.named_children():
                if not hasattr(block, name): continue

                s_child = getattr(block, name)
                if isinstance(m_child, nn.Conv2d):
                    if m_child.weight.shape == s_child.weight.shape:
                        del s_child.weight
                        s_child.weight = m_child.weight

                        if not hasattr(s_child, 'scaler'):
                            s_child.scaler = nn.Parameter(torch.tensor(1.0).to(m_child.weight.device))

                        def new_forward(self, x):
                            return self._conv_forward(x, self.weight * self.scaler, self.bias)
                        
                        s_child.forward = types.MethodType(new_forward, s_child)

# apply_weight_sharing_resnet(model)




