import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-proportion', type=float, default=1.0) # Range between 0 and 1. If you want to use 20% of the training dataset, set this value to be 0.2
    parser.add_argument('--adv-train', action='store_true') # Adversarial
    parser.add_argument('--dis-lambda', type=float, default=1e-2) # % of penalty that adversarial discriminator has on qa loss
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--train-datasets', type=str, default='squad,nat_questions')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--OOD-train-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--OOD-train-dir', type=str, default='datasets/oodomain_train') # OOD training dataset for finetuning
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--OOD-val-dir', type=str, default='datasets/oodomain_val') # OOD validation dataset for finetuning
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test') 
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true') # Train
    parser.add_argument('--do-finetune', action='store_true') # Finetune
    parser.add_argument('--do-eval', action='store_true') # Evaluate
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    return args
