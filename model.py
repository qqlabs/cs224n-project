import torch
import torch.nn as nn
import torch.nn.functional as F

# This creates the Domain Discriminator model
# This model takes in the hidden representation of the question-para pair from the QA model and tries to guess the domain (dataset)
class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=3, input_size=768, # Num classes is 3 - we train the DD on the original 3 ID datasets and then finetune it on the 3 OOD datasets
                 hidden_size=768, num_layers=3, dropout=0.1): # Right now, use 3 hidden layers
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            )) # Domain discriminator is a simple FCN
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x) # The output of the last layer is my logits
        log_prob = F.log_softmax(logits, dim=1) # Pump it into a softmax function
        return log_prob
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # This is the combined QA + Domain Discriminator model    
# class DomainQA(nn.Module):
#     def __init__(self, num_classes=3, hidden_size=768,
#                  num_layers=3, dropout=0.1, dis_lambda=0.5, anneal=False):
#         super(DomainQA, self).__init__()
        
#         self.qa_outputs = nn.Linear(hidden_size, 2)
#         # Initialize weights
#         self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
#         self.qa_outputs.bias.data.zero_()
        
#         input_size = hidden_size
        
#         self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout) # Initialize discriminator

#         self.num_classes = num_classes # Yup
#         self.dis_lambda = dis_lambda # This is the weight we apply to the domain-invariance penalty
#         self.anneal = anneal # LR annealing?
#         self.sep_id = 102

#     # only for prediction
#     def forward(self, input_ids, token_type_ids, attention_mask,
#                 start_positions=None, end_positions=None, labels=None,
#                 dtype=None, global_step=22000):
#         if dtype == "qa":
#             qa_loss = self.forward_qa(input_ids, token_type_ids, attention_mask,
#                                       start_positions, end_positions, global_step)
#             return qa_loss

#         elif dtype == "dis":
#             assert labels is not None
#             dis_loss = self.forward_discriminator(input_ids, token_type_ids, attention_mask, labels)
#             return dis_loss

#         else:
#             sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#             logits = self.qa_outputs(sequence_output)
#             start_logits, end_logits = logits.split(1, dim=-1)
#             start_logits = start_logits.squeeze(-1)
#             end_logits = end_logits.squeeze(-1)

#             return start_logits, end_logits

#     def forward_qa(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions, global_step):
#         sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         cls_embedding = sequence_output[:, 0]
#         if self.concat:
#             sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
#             hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
#         else:
#             hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation
#         log_prob = self.discriminator(hidden)
#         targets = torch.ones_like(log_prob) * (1 / self.num_classes)
#         # As with NLLLoss, the input given is expected to contain log-probabilities
#         # and is not restricted to a 2D Tensor. The targets are given as probabilities
#         kl_criterion = nn.KLDivLoss(reduction="batchmean") # Domain-invariance penalty
#         if self.anneal:
#             self.dis_lambda = self.dis_lambda * kl_coef(global_step)
#         kld = self.dis_lambda * kl_criterion(log_prob, targets)

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         # If we are on multi-GPU, split add a dimension
#         if len(start_positions.size()) > 1:
#             start_positions = start_positions.squeeze(-1)
#         if len(end_positions.size()) > 1:
#             end_positions = end_positions.squeeze(-1)
#         # sometimes the start/end positions are outside our model inputs, we ignore these terms
#         ignored_index = start_logits.size(1)
#         start_positions.clamp_(0, ignored_index)
#         end_positions.clamp_(0, ignored_index)

#         loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
#         start_loss = loss_fct(start_logits, start_positions)
#         end_loss = loss_fct(end_logits, end_positions)
#         qa_loss = (start_loss + end_loss) / 2
#         total_loss = qa_loss + kld
#         return total_loss

#     def forward_discriminator(self, input_ids, token_type_ids, attention_mask, labels):
#         with torch.no_grad():
#             sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#             cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
#             hidden = cls_embedding
#         log_prob = self.discriminator(hidden.detach())
#         criterion = nn.NLLLoss()
#         loss = criterion(log_prob, labels)

#         return loss

#     def get_sep_embedding(self, input_ids, sequence_output):
#         batch_size = input_ids.size(0)
#         sep_idx = (input_ids == self.sep_id).sum(1)
#         sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
#         return sep_embedding
    