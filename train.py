import os
from collections import OrderedDict
import torch

import util

from transformers import AdamW
from tensorboardX import SummaryWriter

from model import DomainDiscriminator

from tqdm import tqdm


#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        # Set parameters
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)
    
    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores

# This implements the QA model with adversarial learning    
class AdversarialTrainer(Trainer):
    def __init__(self, args, log):
        super(AdversarialTrainer, self).__init__(args, log)
        self.n_train_datasets = len(args.train_datasets.split(",")) # This gives me the number of training datasets I have...
        
    def save(self, model):
        torch.save(model.state_dict(), os.path.join(self.save_dir, 'checkpoint_QA')) # Save the QA model
        torch.save(self.Discriminator.state_dict(), os.path.join(self.save_dir, 'checkpoint_discriminator')) # Saves my discriminator model in a discriminator subfolder lol
            
    def create_discriminator(self):
        self.Discriminator = DomainDiscriminator() # Create my discriminator
        self.dis_optim = AdamW(self.Discriminator.parameters(), lr=self.lr) # In this case I am using the same LR as normal QA model
    
    def discriminator_loss(self, dis_log_probs, true_labels):
        criterion = torch.nn.NLLLoss()
        loss = criterion(dis_log_probs, true_labels)
        return loss
        
    # def domain_invariance_penalty(self, outputs):
    #     hidden_states = hidden_states[:, 0] # CLS embeddings
    #     log_prob = self.Discriminator(hidden_states) # Spits out my probabilities
    #     targets = torch.ones_like(log_prob) * (1 / self.n_train_datasets) # Ok so this is what I would get if it was a simple uniform distribution
    #     discriminator_loss = torch.nn.KLDivLoss(reduction="batchmean")(log_prob, targets)
    #     return discriminator_loss
    
    def train(self, qa_model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        qa_model.to(device)
        qa_optim = AdamW(qa_model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    qa_optim.zero_grad()
                    qa_model.train()
                    self.Discriminator.train()
                    
                    # DATA PARSING
                    # Process the data & get the outputs!
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    domain_id = batch['domain_id'].to(device)

                    # QA PREDICTION
                    # First make a QA prediction
                    qa_outputs = qa_model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    output_hidden_states=True)                    
                    # Store the QA loss to train later
                    qa_loss = qa_outputs.loss

                    # QA hidden states
                    # Last layer of hidden_states is pulled like this
                    # See https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/distilbert/modeling_distilbert.py#L330
                    qa_last_hidden_state = qa_outputs.hidden_states[-1]


                    # DISCRIMINATOR TRAINING
                    self.dis_optim.zero_grad() # Set to 0 grad before commencing training

                    # Predict with Discriminator
                    dis_output = self.Discriminator(qa_last_hidden_state)
                    
                    # Get Discriminator Loss
                    dis_loss = self.discriminator_loss(dis_output, domain_id)
                    dis_loss.backward() # Backward propagate
                    self.dis_optim.step() # Take a step
                    self.dis_optim.zero_grad() # Reset to 0 grad
                    

                    # QA TRAINING WITH DISCRIMINATOR LOSS
                    # we subtract the discriminator loss to penalize the qa_loss
                    # since higher loss is "better" (negative loss)
                    total_loss = qa_loss - self.lambda_adv*dis_loss 
                    total_loss.backward()
                    
                    qa_optim.step()
                    qa_optim.zero_grad()
                    
                    
                    # Display epochs and losses in progress bar
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, 
                                             qa_loss=qa_loss.item(),
                                             dis_loss=dis_loss.item(),
                                             total_loss=total_loss.item())

                    # Add losses to tensorboard
                    tbx.add_scalar('train/qa_loss', qa_loss.item(), global_idx)
                    tbx.add_scalar('train/dis_loss', dis_loss.item(), global_idx)
                    tbx.add_scalar('train/total_loss', total_loss.item(), global_idx)

                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(qa_model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(qa_model)
                    global_idx += 1
        return best_scores      
    



