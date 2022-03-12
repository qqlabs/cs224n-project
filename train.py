import os
from collections import OrderedDict
import torch

import util
from util import kl_coef

from transformers import AdamW
from tensorboardX import SummaryWriter

from model import DomainDiscriminator

from tqdm import tqdm

import numpy as np


#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        # Set parameters
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint') # Where I store my model after training it on IID training
        self.finetune_path = os.path.join(args.save_dir, args.finetune_name + '_finetune_checkpoint') # Where I store my model after finetuning it on OOD training
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.finetune_path):
            os.makedirs(self.finetune_path)

    def save(self, model, stage):
        if stage == "train": # Save in different checkpoint folders depending if I'm in train or finetuning stage
            model.save_pretrained(self.path)
        elif stage == "finetune":
            model.save_pretrained(self.finetune_path)
    
    def evaluate(self, models, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        start_logits_all_models = []
        end_logits_all_models = []

        for model in models:
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

            
            start_logits = torch.cat(all_start_logits).cpu().numpy()
            end_logits = torch.cat(all_end_logits).cpu().numpy()
            start_logits_all_models.append(start_logits)
            end_logits_all_models.append(end_logits)

        start_logits = np.mean(start_logits_all_models, axis=0)
        end_logits = np.mean(end_logits_all_models, axis=0)

        # Get F1 and EM scores
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
    
    def train(self, model, train_dataloader, eval_dataloader, val_dict, best_scores, stage):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
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
                        preds, curr_score = self.evaluate([model], eval_dataloader, val_dict, return_preds=True)
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
                            self.save(model, stage) # Where I save depends on which stage I'm in
                    global_idx += 1
        return best_scores

# This implements the QA model with adversarial learning    
class AdversarialTrainer(Trainer):
    def __init__(self, args, log):
        super(AdversarialTrainer, self).__init__(args, log)
        self.dis_lambda = args.dis_lambda
        self.anneal = args.anneal
        self.w_reg = args.w_reg
        self.num_adv_steps = args.num_adv_steps
        self.full_embedding = args.full_embedding
        if self.full_embedding:
            self.input_size = 384 * 768 # Chunk size of 384, each vector is 768 dim
            self.hidden_size = 768 / 16 # Reduce hidden layer size by factor of 16 to improve computational complexity
        else:
            self.input_size = 768 # Pass in CLS token which is 768 dim
            self.hidden_size = 768
        if args.combined or args.combinedwAug:
            if args.binary_align or args.wiki_align:
                self.num_domains = 2
            else:
                self.num_domains = len(args.train_datasets.split(",")) + len(args.OOD_train_datasets.split(","))
        else:
            if args.binary_align:
                self.num_domains = 2
            else:
                self.num_domains = len(args.train_datasets.split(","))
        self.create_discriminator()
        

    def create_discriminator(self):
        self.Discriminator = DomainDiscriminator(num_classes=self.num_domains, input_size=int(self.input_size), hidden_size=int(self.hidden_size))
        self.dis_optim = AdamW(self.Discriminator.parameters(), lr=self.lr) # In this case I am using the same LR as normal QA model
        self.Discriminator.to(self.device)

    def discriminator_loss(self, dis_log_probs, true_labels, loss_type):
        if loss_type == "KLD": # This is the KL Divergence Loss
            targets = torch.ones_like(dis_log_probs) * (1 / self.num_domains) # Simple uniform distribution across number of training datasets
            loss = torch.nn.KLDivLoss(reduction="batchmean")(dis_log_probs, targets)
            return loss

        elif loss_type == "NLL": # This is the negative log likelihood loss
            criterion = torch.nn.NLLLoss()
            loss = criterion(dis_log_probs, true_labels)
            return loss

    def train(self, qa_model, train_dataloader, eval_dataloader, val_dict, best_scores, stage):
        device = self.device
        qa_model.to(device)
        qa_optim = AdamW(qa_model.parameters(), lr=self.lr)
        global_idx = 0
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
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
                    qa_optim.zero_grad()
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

                    if self.full_embedding:
                        # send in inputs for the entire layer
                        qa_hidden_input = torch.flatten(qa_last_hidden_state, start_dim=1)
                    else:
                        # send in the CLS embedding since we have 384 tokens
                        qa_hidden_input = qa_last_hidden_state[:, 0]

                    
                    # Can also average the tokens
                    # qa_hidden_input = torch.mean(qa_last_hidden_state, dim=1)
                    

                    # QA TRAINING WITH DISCRIMINATOR ADVERSAIRAL LOSS
                    # we add the KL adversarial loss to the normal CE loss
                    
                    # Call discriminator to compute the KL adversarial loss
                    adv_output = self.Discriminator(qa_hidden_input)
                    KL_adv_loss = self.discriminator_loss(adv_output, domain_id, "KLD")

                    # This is the new loss that penalizes the QA loss if adv_loss does well
                    if self.anneal:
                        self.dis_lambda = 0.01 * kl_coef(global_idx) # Anneal accordingly (intuition: progressively train discriminator with harder examples)

                    total_loss = qa_loss + self.dis_lambda*KL_adv_loss
                    total_loss = total_loss.mean()
                    total_loss.backward()
                    
                    qa_optim.step()



                    # As per the original GAN algorithm, let the discriminator train multiple times for a given batch
                    # Give the discriminator a better fighting chance
                    for step in range(self.num_adv_steps):
                        # Impose W Regularization if needed
                        # This clips our weights to enforce gradient constraint for smoother training of discriminator
                        if self.w_reg:
                            for param in self.Discriminator.parameters():
                                param.data.clamp_(-0.01, 0.01) # This is the suggested range used in the WGAN paper
                                
                        self.dis_optim.zero_grad()
                        dis_output = self.Discriminator(qa_hidden_input.clone().detach())
                        dis_loss = self.discriminator_loss(dis_output, domain_id, "NLL")
                        dis_loss = dis_loss.mean() # average the loss across batch
                        dis_loss.backward() # Backward propagate
                        self.dis_optim.step() # Take a step
                    
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
                        preds, curr_score = self.evaluate([qa_model], eval_dataloader, val_dict, return_preds=True)
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
                            self.save(qa_model, stage)
                    global_idx += 1
        return best_scores      
    



