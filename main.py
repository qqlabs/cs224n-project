from collections import defaultdict
from collections import OrderedDict
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

        # Train on IID + OOD
        if args.combined or args.combinedwAug: 
            if args.binary_align: # Domain ID is binary - 0 for IID and 1 for OOD
                # Create cache by tokenizing, don't load anything so we use less memory
                # Note, here the domain ID will be 0
                for dataset_name in args.train_datasets.split(','):
                    create_cache(args, dataset_name, args.train_dir, tokenizer, 'train', 0)

                # For this, the domain ID will be 1
                for dataset_name in args.OOD_train_datasets.split(','):
                    if args.combined:
                        create_cache(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', 1)
                    else:
                        create_cache(args, dataset_name + "_combined", args.OOD_train_dir, tokenizer, 'train', 1)                     

                train_dataset = []
                for dataset_name in args.train_datasets.split(','):
                    tmp_train_dataset, _ = get_dataset(args, dataset_name, args.train_dir, tokenizer, 'train', 0)
                    train_dataset.append(tmp_train_dataset)

                for dataset_name in args.OOD_train_datasets.split(','):
                    if args.combined:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', 1)
                    else:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name + "_combined", args.OOD_train_dir, tokenizer, 'train', 1)
                    train_dataset.append(tmp_train_dataset)

            elif args.wiki_align:
                if args.combined:
                    # Wiki
                    create_cache(args, 'squad', args.train_dir, tokenizer, 'train', 0)    
                    create_cache(args, 'nat_questions', args.train_dir, tokenizer, 'train', 0)
                    create_cache(args, 'relation_extraction', args.OOD_train_dir, tokenizer, 'train', 0)
                    # Non-Wiki                            
                    create_cache(args, 'newsqa', args.train_dir, tokenizer, 'train', 1)    
                    create_cache(args, 'duorc', args.OOD_train_dir, tokenizer, 'train', 1)
                    create_cache(args, 'race', args.OOD_train_dir, tokenizer, 'train', 1)       

                else:
                    # Wiki
                    create_cache(args, 'squad', args.train_dir, tokenizer, 'train', 0)    
                    create_cache(args, 'nat_questions', args.train_dir, tokenizer, 'train', 0)
                    create_cache(args, 'relation_extraction_combined', args.OOD_train_dir, tokenizer, 'train', 0)
                    # Non-Wiki                            
                    create_cache(args, 'newsqa', args.train_dir, tokenizer, 'train', 1)    
                    create_cache(args, 'duorc_combined', args.OOD_train_dir, tokenizer, 'train', 1)
                    create_cache(args, 'race_combined', args.OOD_train_dir, tokenizer, 'train', 1)                                     

                train_dataset = []
                for dataset_name in args.train_datasets.split(','):
                    if dataset_name == "newsqa":
                        domain_id = 1
                    else:
                        domain_id = 0
                    tmp_train_dataset, _ = get_dataset(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
                    train_dataset.append(tmp_train_dataset)

                for dataset_name in args.OOD_train_datasets.split(','):
                    if dataset_name == "relation_extraction":
                        domain_id = 0
                    else:
                        domain_id = 1
                    if args.combined:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', domain_id)
                    else:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name + "_combined", args.OOD_train_dir, tokenizer, 'train', domain_id)
                    train_dataset.append(tmp_train_dataset)

            else: # Standard multi-source alignment. Each dataset gets an index from 0 to 5
                # Note, here the domain ID will be from 0 to 2
                for domain_id, dataset_name in enumerate(args.train_datasets.split(',')):
                    create_cache(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
         
                num_IID_dataset = len(args.train_datasets.split(','))            
                # For this, the domain ID should go from 3 to 5
                for domain_id, dataset_name in enumerate(args.OOD_train_datasets.split(',')):
                    if args.combined:
                        create_cache(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', domain_id+num_IID_dataset)
                    else:
                        create_cache(args, dataset_name + "_combined", args.OOD_train_dir, tokenizer, 'train', domain_id+num_IID_dataset)                      

                train_dataset = []

                for domain_id, dataset_name in enumerate(args.train_datasets.split(',')):
                    tmp_train_dataset, _ = get_dataset(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
                    train_dataset.append(tmp_train_dataset)

                for domain_id, dataset_name in enumerate(args.OOD_train_datasets.split(',')):
                    if args.combined:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name, args.OOD_train_dir, tokenizer, 'train', domain_id+num_IID_dataset)
                    else:
                        tmp_train_dataset, _ = get_dataset(args, dataset_name + "_combined", args.OOD_train_dir, tokenizer, 'train', domain_id+num_IID_dataset)
                    train_dataset.append(tmp_train_dataset)

        else: # Train on only IID
            if args.wiki_align: # This is if I use wiki alignment on only IID
                # Wiki
                create_cache(args, 'squad', args.train_dir, tokenizer, 'train', 0)    
                create_cache(args, 'nat_questions', args.train_dir, tokenizer, 'train', 0)
                # Non-Wiki                            
                create_cache(args, 'newsqa', args.train_dir, tokenizer, 'train', 1)                                

                train_dataset = []
                for dataset_name in args.train_datasets.split(','):
                    if dataset_name == "newsqa":
                        domain_id = 1
                    else:
                        domain_id = 0
                    tmp_train_dataset, _ = get_dataset(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
                    train_dataset.append(tmp_train_dataset)

            else: # Standard IID datasets without special alignment
                for domain_id, dataset_name in enumerate(args.train_datasets.split(',')):
                    create_cache(args, dataset_name, args.train_dir, tokenizer, 'train', domain_id)
                train_dataset = []
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

        # Grab IID validation datasets
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
            # combine variants if needed
            if args.variants == '':
                util.write_squad(util.read_squad(f'{args.OOD_train_dir}/{dataset_name}_orig'), f'{args.OOD_train_dir}/{dataset_name}')
            else:
                util.combine_qas(f'{args.OOD_train_dir}/{dataset_name}', args.variants.split(','), with_suffix=False)

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

        # Load QA trainer    
        trainer = Trainer(args, log)

        # Now, finetune my model
        best_scores = {'F1': -1.0, 'EM': -1.0}    
        best_scores_finetune = trainer.train(model, OOD_train_loader, OOD_val_loader, OOD_val_dict, best_scores, "finetune")

    
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        

        finetune_names = args.finetune_name.split(',')

        models = []

        save_dirs = args.save_dir.split(',')

        log = util.get_logger(save_dirs[0], f'log_{split_name}')
        trainer = Trainer(args, log)

        for idx, finetune_name in enumerate(finetune_names):
            if finetune_name == 'none':
                checkpoint_path = os.path.join(save_dirs[idx], 'checkpoint')
            else:
                checkpoint_path = os.path.join(save_dirs[idx], finetune_name + '_finetune_checkpoint') # Load the FINETUNED model. Note: we should add a toggle here...
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
            model.to(args.device)
            models.append(model)

        # evaluate on every dataset in eval_dir
        eval_datasets = [f for f in os.listdir(args.eval_dir) if ".pt" not in f]
        # combined_eval_data = ','.join(map(str, eval_datasets)) # also eval over all combined datasets
        # eval_datasets.append(combined_eval_data)

        num_qas = {}
        eval_scores_dict = {}
        eval_scores_dict['Overall'] = defaultdict(int)

        all_preds = OrderedDict()
        all_gold = {'question': [], 'context': [], 'id': [], 'answer': []}

        for dataset in eval_datasets:
            eval_dataset, eval_dict = get_dataset(args, dataset, args.eval_dir, tokenizer, split_name)
            all_gold['question'].extend(eval_dict['question'])
            all_gold['context'].extend(eval_dict['context'])
            all_gold['id'].extend(eval_dict['id'])
            all_gold['answer'].extend(eval_dict['answer'])

            num_qas[dataset] = len(eval_dict['question'])
            eval_loader = DataLoader(eval_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(eval_dataset))
            eval_preds, eval_scores = trainer.evaluate(models, eval_loader,
                                                    eval_dict, return_preds=True,
                                                    split=split_name)
            all_preds.update(eval_preds)
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
        log.info(f'Finetune {args.finetune_name} Scores: {results_str}')

        # calculate F1 per row to find mistakes
        output_dict = util.error_analysis(all_gold, all_preds)

        # Write error analysis
        if args.error_file != "":
            sub_path = os.path.join(save_dirs[0], split_name + '_' + args.error_file)            
            log.info(f'Writing error analysis file to {sub_path}...')
            with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
                csv_writer = csv.writer(csv_fh, delimiter=',')
                csv_writer.writerow(['Id', 'Predicted', 'Gold', 'F1', 'EM'])
                for uuid in sorted(output_dict):
                    csv_writer.writerow([uuid, output_dict[uuid]['pred'], output_dict[uuid]['gold'], output_dict[uuid]['f1'], output_dict[uuid]['em']])

        # Write submission file
        if args.sub_file != "":
            sub_path = os.path.join(save_dirs[0], split_name + '_' + args.sub_file)            
            log.info(f'Writing submission file to {sub_path}...')
            with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
                csv_writer = csv.writer(csv_fh, delimiter=',')
                csv_writer.writerow(['Id', 'Predicted'])
                for uuid in sorted(output_dict):
                    csv_writer.writerow([uuid, output_dict[uuid]['pred']])


if __name__ == '__main__':
    main()