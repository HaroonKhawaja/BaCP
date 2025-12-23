import torch
import torch.nn as nn

class DyReLUB(nn.Module):
    def __init__(self, channels, K=2, reduction=4):
        super().__init__()
        self.channels = channels
        self.K = K

        self.hyperfunction = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, 2*K*channels),
            nn.Sigmoid()
        )

        self.register_buffer('lambdas', torch.tensor([1.0, 0.5]))
        self.register_buffer('a_init', torch.tensor([1.0] + [0.0] * (K - 1))) 
        self.register_buffer('b_init', torch.tensor([0.0] + [0.0] * (K - 1)))


    def get_relu_coefs(self, x):
        B, C, H, W = x.shape
        theta = x.mean(dim=(2, 3))          # (B, C)

        theta = self.hyperfunction(theta)   # (B, 2*K*C)
        theta = theta * 2 - 1               # Normalize to [-1, 1]

        theta = theta.view(B, C, 2*self.K)  # (B, C, 2K)
        return theta

    def forward(self, x):
        theta = self.get_relu_coefs(x)

        theta_a = theta[:, :, :self.K]
        theta_b = theta[:, :, self.K:]

        a = theta_a * self.lambdas[0] + self.a_init
        b = theta_b * self.lambdas[1] + self.b_init

        a = a.view(theta.shape[0], self.channels, self.K, 1, 1)
        b = b.view(theta.shape[0], self.channels, self.K, 1, 1)

        x_expanded = x.unsqueeze(2)         # (B, C, 1, H, W)
        out = a * x_expanded + b            # (B, C, K, H, W)
        
        # Maxing over K
        out, _ = out.max(dim=2)             # (B, C, H, W)
        return out

class DyReLUAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dyrelu = DyReLUB(channels)
        self.relu = nn.ReLU()

        self.t_start = None
        self.t_end = None
        self.t = 0
        self.current_beta = 1.0

    def get_beta(self):
        if self.t_start is None or self.t_end is None:
            raise Exception("Please set t_start and t_end")
        
        if self.t < self.t_start:
            return 1.0
        elif self.t_start <= self.t <= self.t_end:
            return 1.0 - ((self.t - self.t_start) / (self.t_end - self.t_start))
        else:
            return 0.0
        
    def forward(self, x):
        beta = self.get_beta()

        if beta == 0.0:
            return self.relu(x)
        if beta == 1.9:
            return self.dyrelu(x)
        
        return beta * self.dyrelu(x) + (1 - beta) * self.relu(x)

    def step(self):
        self.t += 1
        self.current_beta = self.get_beta()

# Helper functions
def set_t_for_dyrelu_adapter(model, t_start, t_end):
    print(f'[DyReLU ADAPTER] Setting t_start={t_start}, t_end={t_end}')
    for m in model.modules():
        if isinstance(m, DyReLUAdapter):
            m.t_start = t_start
            m.t_end = t_end

def step_dyrelu_adapter(model):
    beta_val = None
    for m in model.modules():
        if isinstance(m, DyReLUAdapter):
            m.step()
            beta = m.get_beta()
            beta_val = beta
    print(f'[DyReLU ADAPTER] Updated Beta: {beta_val:.4f}')






