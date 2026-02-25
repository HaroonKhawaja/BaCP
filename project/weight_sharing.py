import torch
import torch.nn as nn
import torch.nn.functional as F
import types

def apply_weight_sharing_resnet(model_wrapper, R=2):
    model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    model.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    master_idx = R - 1
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in layer_names:
        if not hasattr(model, layer_name): continue

        layer_container = getattr(model, layer_name)
        num_blocks = len(layer_container)

        if num_blocks <= R: continue

        master_block = layer_container[master_idx]
        for i in range(master_idx + 1, num_blocks):
            block = layer_container[i]

            for name, m_child in master_block.named_children():
                if not hasattr(block, name): continue

                s_child = getattr(block, name)
                if isinstance(m_child, nn.Conv2d) and isinstance(s_child, nn.Conv2d):
                    if m_child.weight.shape == s_child.weight.shape:
                        del s_child.weight
                        s_child.weight = m_child.weight

                        if not hasattr(s_child, 'scaler'):
                            s_child.scaler = nn.Parameter(torch.tensor(1.0).to(m_child.weight.device))

                        def new_forward(self, x):
                            scaled_w = self.weight * self.scaler
                            return F.conv2d(
                                x, scaled_w, self.bias, 
                                self.stride, self.padding, self.dilation, self.groups
                            )
                        
                        s_child.forward = types.MethodType(new_forward, s_child)

    model.unique_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[WEIGHT SHARING] Sharing Applied.")
    print(f" > Theoretical Params: {model.total_params:,}")
    print(f" > Unique Params:      {model.unique_params:,}")
    print(f" > Intrinsic Sparsity: {1.0 - (model.unique_params/model.total_params):.2%}")
    
# apply_weight_sharing_resnet(model)




