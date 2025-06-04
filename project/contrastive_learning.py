import torch
from loss_fn import SupConLoss, NTXentLoss
from utils import get_device
from tqdm import tqdm
from utils import *
from constants import *

class ContrastiveLearner(object):
    """
    Implements contrastive learning training for both supervised and unsupervised approaches.
    Supports SupConLoss (supervised) and NT-XentLoss (unsupervised) contrastive losses.
    """
    def __init__(self, model, config):
        self.device = get_device()
        self.model = model.to(self.device)

        # Required configuration parameters
        self.optimizer = config['optimizer']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']

        # Optional configuration parameters with defaults
        self.scheduler = config.get('scheduler', None)
        self.n_views = config.get('n_views', 2)
        self.temperature = config.get('temperature', TEMP)
        self.base_temperature = config.get('base_temperature', BASE_TEMP)
        self.loss_type = config.get('loss_type', 'supcon')
        self.epsilon = 1e-6
        
        # Loss functions
        self.supervised_loss = SupConLoss(self.n_views, self.temperature, self.base_temperature, self.batch_size)
        self.unsupervised_loss = NTXentLoss(self.n_views, self.temperature)

    # Initialize loss functions
    def supervised_criterion(self, features, labels):
        """
        Calculates supervised contrastive loss.
        Pulls together embeddings from the same class while pushing apart embeddings from different classes.
        """
        loss = self.supervised_loss(features, labels)
        return loss
    
    def unsupervised_criterion(self, features1, features2):
        """
        Calculates unsupervised contrastive loss in both directions.
        Averaging the bidirectional loss ensures symmetry in the learned representations.
        """
        forward_loss = self.unsupervised_loss(features1, features2)
        backward_loss = self.unsupervised_loss(features2, features1)
        total_loss = (forward_loss + backward_loss) / 2
        return total_loss
        
    def train(self, train_loader, save_path, logger=None):
        """
        Main training loop for contrastive learning.
        """
        # Initializing logger and logging hyper-parameters
        if logger is not None:
            logger.create_log()
            logger.log_hyperparameters({
                'epochs': self.epochs, 
                'n_views': self.n_views, 
                'optimizer': self.optimizer, 
                'scheduler': self.scheduler, 
                'batch_size': self.batch_size, 
                'temperature': self.temperature, 
                'base_temperature': self.base_temperature
                })
        
        epoch_avg_losses = []
        for epoch in range(self.epochs):
            batch_losses = []
            batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for idx, (images, labels) in enumerate(batch):
                # Extracting different augmented views of the same image and moving them to device
                images1, images2 = images[0], images[1]
                images1, images2, labels = images1.to(self.device), images2.to(self.device), labels.to(self.device)
                
                # Skipping incomplete batches
                if images1.size(0) != images2.size(0) or images1.size(0) < self.batch_size:
                    continue
                
                # Generating feature embeddings in both views
                features1 = self.model(images1)
                features2 = self.model(images2)
                
                # Calculating appropriate loss based on selected loss type
                if self.loss_type == "unsupcon":
                    # Unsupervised contrastive learning
                    loss = self.unsupervised_criterion(features1, features2)

                elif self.loss_type == "supcon":
                    # Supervised contrastive learning
                    features = torch.cat([features1, features2], dim=0)
                    loss = self.supervised_criterion(features, labels)

                else:
                    # Hybrid approach: combines both supervised and unsupervised losses
                    features = torch.cat([features1, features2], dim=0)
                    loss_sup = self.supervised_criterion(features, labels)
                    loss_unsup = self.unsupervised_criterion(features1, features2)
                    loss = (loss_sup + loss_unsup)

                batch_losses.append(loss)
                
                # Back-propagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch.set_postfix({"Loss": f"{loss:.3f} "})
            
            # Updating scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Averaging batch losses for the epoch
            epoch_loss = torch.mean(torch.tensor(batch_losses))
            epoch_avg_losses.append(epoch_loss)

            # Logging epoch results
            info = f"Epoch [{epoch + 1}/{self.epochs}]: Average Loss: {epoch_loss:.5f}\n" 
            print(info)

            if logger is not None:
                logger.log_epochs(info)       
            
            # Saving model weights
            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)

        return epoch_avg_losses
