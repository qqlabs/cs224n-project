from collections import defaultdict
import os
import csv
import json
import util
import pickle

from args import get_train_test_args

import torch
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from train import Trainer, AdversarialTrainer

from data_processing import create_cache, get_dataset


def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    # Load up DistilBert
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Choose between normal QA model or QA with adversarial
        if args.adv_train:
            trainer = AdversarialTrainer(args, log)
        else:
            trainer = Trainer(args, log)
            
        train_dataset = []

        # Create cache by tokenizing, don't load anything so we use less memory
        for domain_id, dataset_name in enumerate(args.train_datasets.split(',')):
            create_cache(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)

        # We need to include the domain_id in our training data since the gan needs the 
        # true domain_id to calculate loss.
        # We will assign each domain_id based on the order they are listed in the
        # args for training datasets. This will remain fixed.

        # We only include domain_id in training data since the gan only operates on the
        # training stage.
        for domain_id, dataset_name in enumerate(args.train_datasets.split(',')):
            tmp_train_dataset, _ = get_dataset(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
            train_dataset.append(tmp_train_dataset)

        # Concat my datasets together
        train_set = ConcatDataset(train_dataset)
        # This allows you to take a smaller subset of the training dataset if you want to quickly test out stuff
        # The default value for sample_proportion is 1 (i.e. you train the model on the entire training dataset)
        sample_index = list(range(0, len(train_set), int(1/args.sample_proportion)))        
        train_set = torch.utils.data.Subset(train_set, sample_index) # Grab my subset

        train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                # sampler=RandomSampler(train_dataset),
                                shuffle=True)

        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        # Train on IID datasets
        best_scores = {'F1': -1.0, 'EM': -1.0}                        
        best_scores = trainer.train(model, train_loader, val_loader, val_dict, best_scores, "train")

        # Save my best score
        pickle.dump(best_scores, open(args.save_dir + "/best_scores.p", "wb"))

    if args.do_finetune:
        log = util.get_logger(args.save_dir, 'log_finetune')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Now, prepare my OOD datasets
        log.info("Preparing OOD Training Data...")
        OOD_train_dataset = []

        for domain_id, dataset_name in enumerate(args.OOD_train_datasets.split(',')):
            create_cache(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', domain_id)
 
        for domain_id, dataset_name in enumerate(args.OOD_train_datasets.split(',')):
            tmp_train_dataset, _ = get_dataset(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', domain_id)
            OOD_train_dataset.append(tmp_train_dataset)

        OOD_train_loader = DataLoader(ConcatDataset(OOD_train_dataset),
                                batch_size=args.batch_size,
                                # sampler=RandomSampler(train_dataset),
                                shuffle=True)

        log.info("Preparing OOD Validation Data...")
        OOD_val_dataset, OOD_val_dict = get_dataset(args, args.OOD_train_datasets, args.OOD_val_dir, tokenizer, 'val')
        
        OOD_val_loader = DataLoader(OOD_val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(OOD_val_dataset))   

        # Load my last checkpoint
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint') # Find my checkpoint
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path) # Load checkpoint
        log.info("Model loaded from " + checkpoint_path)

        # Load my last set of best scores
        # best_scores = pickle.load(open(args.save_dir + "/best_scores.p", "rb"))
        # log.info("Previous best score loaded!")
        # log.info(str(best_scores))

        # Choose between normal QA model or QA with adversarial       
        trainer = Trainer(args, log)

        # Now, finetune my model
        best_scores = {'F1': -1.0, 'EM': -1.0}    
        best_scores_finetune = trainer.train(model, OOD_train_loader, OOD_val_loader, OOD_val_dict, best_scores, "finetune")
        # FYI: The best scores that I load into here is the final best scores from the first part of the training
        # I.e. I will not update my model's parameters unless it beats the previous results

        # Save my best score
        # pickle.dump(best_scores_finetune, open(args.save_dir + "finetuned_best_scores.p", "wb"))
    
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'finetune_checkpoint') # Load the FINETUNED model
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        # evaluate on every dataset in eval_dir
        eval_datasets = [f for f in os.listdir(args.eval_dir) if ".pt" not in f]
        # combined_eval_data = ','.join(map(str, eval_datasets)) # also eval over all combined datasets
        # eval_datasets.append(combined_eval_data)

        num_qas = {}
        eval_scores_dict = {}
        eval_scores_dict['Overall'] = defaultdict(int)

        for dataset in eval_datasets:
            eval_dataset, eval_dict = get_dataset(args, dataset, args.eval_dir, tokenizer, split_name)
            num_qas[dataset] = len(eval_dict['question'])
            eval_loader = DataLoader(eval_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(eval_dataset))
            eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                    eval_dict, return_preds=True,
                                                    split=split_name)
            eval_scores_dict[dataset] = eval_scores
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())

            log.info(f'{dataset} Eval {results_str}')

        for dataset in eval_datasets:
            for k, v in eval_scores_dict[dataset].items():
                eval_scores_dict['Overall'][k] += num_qas[dataset]/sum(num_qas.values())*v
        
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores_dict['Overall'].items())
        log.info(f'Overall Eval {results_str}')

        results_str = ''
        dataset_str = ''
        for k,v in eval_scores_dict.items():
            dataset_str += k + '\t'
            for metric, score in v.items():
                results_str += f'{score:05.2f}' + '\t'

        log.info('Easy Copy Paste')
        log.info(f'Datasets: {dataset_str}')
        log.info(f'Scores: {results_str}')

        # Write submission file
        if args.sub_file != "":
            sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)            
            log.info(f'Writing submission file to {sub_path}...')
            with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
                csv_writer = csv.writer(csv_fh, delimiter=',')
                csv_writer.writerow(['Id', 'Predicted'])
                for uuid in sorted(eval_preds):
                    csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()