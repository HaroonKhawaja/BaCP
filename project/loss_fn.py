import torch
import torch.nn as nn
from utils import *

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss implementation.
    Pulls together embeddings of samples from the same class while pushing embeddings of samples from different classes apart.
    """
    def __init__(self, n_views, temperature, base_temperature, batch_size):
        super(SupConLoss, self).__init__()
        self.n_views = n_views                  
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.batch_size = batch_size
        self.device = get_device()
        self.eps = 1e-8
        
    def forward(self, features, labels):
        # Total number of feature vectors (batch_size * n_views)
        multiview_batch_size = features.shape[0]
        batch_size = multiview_batch_size // self.n_views
        
        # Duplicate labels to match the multiple views of each sample
        labels = labels.view(-1, 1)
        labels = labels.repeat(self.n_views, 1)

        # Creating mask where mask[i, j] = 1 if labels[i] == labels[j] and 0 otherwise
        positive_pair_mask = labels.eq(labels.T).float().to(self.device)

        # Removing self-similarities (diagonal entries) from positive pairs
        positive_pair_mask =  (1 - torch.eye(positive_pair_mask.shape[0], device=self.device)) * positive_pair_mask

        # Counting positive pairs for each sample (for normalization)
        num_positive_per_sample = positive_pair_mask.sum(1)

        # Avoiding division by zero by setting less than minimum threshold to 1
        num_positive_per_sample = torch.where(num_positive_per_sample < 1e-6, 1, num_positive_per_sample)

        # Creating mask to exclude self-similarity in denominator of contrastive loss
        self_mask_indices = torch.arange(multiview_batch_size).view(-1, 1).to(self.device)
        all_but_self_mask = torch.scatter(torch.ones_like(positive_pair_mask), 1, self_mask_indices, 0)
        
        # Calculating similarity between all samples (dot product of normalized features)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Stabilizing the matrix by subtracting max logit from each row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max
        
        # Computing log probability
        exp_logits = torch.exp(logits) * all_but_self_mask
        exp_logits_sum = torch.clamp(exp_logits.sum(dim=1, keepdim=True), min=self.eps)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_positive = (positive_pair_mask * log_prob).sum(1) / num_positive_per_sample
        
        # Scaling loss by temperature ratio and average across the batch
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_positive
        loss = loss.view(self.n_views, batch_size).mean()
        return loss 

class NTXentLoss(nn.Module):
    """
    Normalized Temperatue-scaled Cross Entropy Loss (NT-Xent Loss).
    Used for unsupervised contrastive learning like SimCLR.
    Treats augmented views of the same sample as positive pairs and different samples as negative pairs.
    """
    def __init__(self, n_views, temperature):
        super(NTXentLoss, self).__init__()
        self.n_views = n_views
        self.temperature = temperature
        self.device = get_device()
        self.eps = 1e-8
    
    def forward(self, features1, features2):
        # Concatenating feature vectors from both views
        features = torch.cat((features1, features2))    # Shape: [2N, 128]
        batch_size = features1.shape[0]    # Shape: N
        
        # Creating labels to identify positve pairs
        sample_indices = torch.arange(batch_size)
        repeated_indices = torch.cat([sample_indices for i in range(self.n_views)])

        # Creating a mask where mask[i, j] = 1 if samples i and j are from different views of the same sample
        labels = repeated_indices.view(-1, 1)  # Shape: [2N, 1]
        positive_pair_mask = labels.eq(labels.T).float().to(self.device)
    
        # Calculating similarity between all samples (dot product of normalized features)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
            
        # Stabilizing the matrix by subtracting max logit from each row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max
        
        # Removing self-similarities (diagonal entries) from positive pairs
        identity_mask = torch.eye(positive_pair_mask.shape[0], device=self.device).bool()

        # Removing diagonal from positive pair mask and similarity matrix
        positive_pair_mask_no_self = positive_pair_mask[~identity_mask].view(positive_pair_mask.shape[0], -1)
        similarity_matrix_no_self = similarity_matrix[~identity_mask].view(similarity_matrix.shape[0], -1)
        
        # Sum exp of similarities for positive pairs
        positive_similarities = similarity_matrix_no_self[positive_pair_mask_no_self.bool()].view(positive_pair_mask.shape[0], -1)
        numerator = torch.sum(torch.exp(positive_similarities), dim=1, keepdim=True)
        
        # Sum exp for all pairs except self-similarities
        denominator = torch.sum(torch.exp(similarity_matrix_no_self), dim=1, keepdim=True)
        denominator = torch.where(denominator < 1e-6, 1, denominator)
        
        # Calculating negative log of (numerator / denominator)
        ratio = numerator / denominator
        ratio = torch.clamp(ratio, min=self.eps, max=1.0)
        log_probability = -torch.log(ratio)

        # Averaging loss across all samples
        loss = torch.mean(log_probability)
        return loss
    

