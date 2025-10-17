import torch
import torch.nn as nn

class DyReLUB(nn.Module):
    def __init__(self, channels, K=2, reduction=4):
        super().__init__()
        self.channels = channels
        self.K = K

        self.hyperfunction = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, 2*K*channels),
            nn.Sigmoid()
        )

        self.register_buffer('lambdas', torch.tensor([1., 0.5]))
        self.register_buffer('v_init', torch.tensor([1.] + [0.]*(K-1)))

    def get_relu_coefs(self, x):
        B, C, H, W = x.shape
        theta = x.mean(dim=(2, 3))          # (B, C)
        theta = self.hyperfunction(theta)   # (B, 2*K*C)
        theta = theta * 2 - 1               # Normalize to [-1, 1]
        theta = theta.view(B, C, 2*self.K)  # (B, C, 2K)
        return theta

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.get_relu_coefs(x)
        a = theta[:, :, :self.K] * self.lambdas[0] + self.v_init[0] # (B, C, K)
        b = theta[:, :, self.K:] * self.lambdas[1] + self.v_init[1] # (B, C, K)

        a = a.unsqueeze(-1).unsqueeze(-1)   # (B, C, K, 1, 1)
        b = b.unsqueeze(-1).unsqueeze(-1)   # (B, C, K, 1, 1)
        x_expanded = x.unsqueeze(2)         # (B, C, 1, H, W)
        out = a * x_expanded + b            # (B, C, K, H, W)
        
        # Maxing over K
        out, _ = out.max(dim=2)             # (B, C, H, W)
        return out

class DyReLUAdapter(nn.Module):
    def __init__(self, t_start, t_end, device='cuda'):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end
        self.t = 0
        self.device = device
        self.dyrelu_cache = {}
        self.activations = {}
        self.call_counts = {}
        self._registered = False

    def get_beta(self):
        if self.t_start <= self.t <= self.t_end:
            return 1 - ((self.t - self.t_start) / (self.t_end - self.t_start))
        else:
            return 0.0
        
    def step(self):
        self.t += 1

    def hook_pre(self, module, input):
        self.activations[module] = input[0].detach()
    
    def hook_post(self, module, input, output):
        beta = self.get_beta()
        activ = self.activations.get(module, None)
        if activ is None:
            return output
        
        m_id = id(module)
        in_channels = activ.shape[1]
        self.call_counts[id(module)] = self.call_counts.get(id(module), 0) + 1
        occurance = self.call_counts[id(module)]
        key = (m_id, in_channels, occurance)
        if key not in self.dyrelu_cache:
            self.dyrelu_cache[key] = DyReLUB(in_channels).to(self.device)

        dyrelu = self.dyrelu_cache[key]
        dy_output = dyrelu(activ)
        return beta * dy_output + (1 - beta) * output

    def register_hooks(self, model):
        if self._registered:
            return
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_pre_hook(self.hook_pre)
                module.register_forward_hook(self.hook_post)
        self._registered = True

    def reset_per_forward(self):
        self.activations.clear()
        self.call_counts.clear()

    def attach_to_model(self, model):
        self.register_hooks(model)
        orig_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            self.reset_per_forward()
            return orig_forward(*args, **kwargs)

        model.forward = wrapped_forward










