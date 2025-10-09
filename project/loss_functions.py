import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temp, base_temp, device, n_views=2, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temp = temp
        self.base_temp = base_temp
        self.device = device
        self.n_views = n_views
        self.eps = eps
    
    def forward(self, z1, z2, labels):
        z = torch.cat([z1, z2], dim=0)
        N_total = z.shape[0]

        labels = labels.view(-1, 1).repeat(self.n_views, 1)

        mask_pos = torch.eq(labels, labels.T).float() 
        diag_mask = torch.eye(N_total, device=z.device)
        mask_pos = mask_pos - diag_mask
        neg_mask = 1 - diag_mask

        logits = torch.matmul(z, z.T) / self.temp

        num_pos = torch.sum(mask_pos, dim=1).clamp(min=1.0)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max

        denom = torch.sum(torch.exp(logits) * neg_mask, dim=1, keepdim=True)

        log_prob = (logits - torch.log(denom + self.eps)) * mask_pos
        mean_log_prob = torch.sum(log_prob, dim=1) / num_pos
        loss = torch.mean(-mean_log_prob)
        return loss
    
class NTXentLoss(nn.Module):
    def __init__(self, temp, device, n_views=2, eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.temp = temp
        self.device = device
        self.n_views = n_views
        self.eps = 1e-8

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        N_total = z.shape[0]
        N = N_total // self.n_views

        mask = torch.eye(N_total, device=z.device).bool()

        logits = torch.matmul(z, z.T) / self.temp

        logits.masked_fill_(mask, float('-inf'))

        targets = torch.arange(N, device=z.device)
        targets = torch.cat([targets + N, targets], dim=0)

        loss = nn.CrossEntropyLoss()(logits, targets)
        return loss



