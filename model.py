import torch
import torch.nn as nn
import torch.nn.functional as F

from args import get_train_test_args

# This creates the Domain Discriminator model
# This model takes in the hidden representation of the question-para pair from the QA model and tries to guess the domain (dataset)
class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768, # Default num classes is 3 - we train the DD on the original 3 ID datasets and then finetune it on the 3 OOD datasets
                 hidden_size=768, num_layers=3, dropout=0.1): # Right now, use 3 hidden layers
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size

            if args.discrim_aug:
                hidden_layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.LeakyReLU() # Implement Leaky ReLU to tackle problem of vanishing gradients. Remove dropout to improve rate of convergence.
                )) # Domain discriminator is a simple FCN

            else:
                 hidden_layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )) 

        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x) # The output of the last layer is my logits
        log_prob = F.log_softmax(logits, dim=1) # Pump it into a softmax function
        return log_prob   
    
    
    
    
    
    
    
    
    
    