# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

## Commands for Data Augmentation

We leverage the nlpaug package that can be found at https://github.com/makcedward/nlpaug for the synonym swapping step, adapted to QA pairs.
We leverage the multitask T5 model that can be found at https://github.com/patil-suraj/question_generation, adapted to work with our dataset.

Synthetic data and augmented data that we generated and used for our experiment can be found in the augmented_synthetic_datasets folder. Simply copy all those files into the oodomain_train dataset folder to replicate our results.

We generated these files using the following commands below:

Synthetic Examples  
python run_aug.py --synth-file datasets/oodomain_train/duorc  
  
Data Augmentation  
python run_aug.py --repeat-aug 1 --aug datasets/oodomain_train/duorc  
  
Combine Synthetic + Augmentation Variants  
python run_aug.py --variants synth,aug --combine datasets/oodomain_train/duorc  
  
## Commands for Training
* Change training datasets with flags  
  * ID: default  
  * ID + OOD: --combined  
  * ID + OOD augmented: --combinedwAug (need to change datasets)  
* Use flag --adv-train for adversarial model  
* Use flag --wiki-align for wiki alignment  

Example: Train with augmented data and synthetic data, wiki aligned, wasserstein regularization, and lambda annealing
python main.py --run-name fancyAdv_wiki_synth_aug1 --do-train --adv-train --combinedwAug --num-epochs 3 --dis-lambda 0.01 --wiki-align --w-reg --anneal

## Commands for Finetuning
* Can finetune on OOD, OOD+synth, OOD+aug, or OOD+synth+aug  
* Use --variants flag to control what sample is being used for finetuning  

Finetuning Example (finetune with both synthetic QA pairs and augmented data)
python main.py --do-finetune --finetune-name synth_aug --variants synth,aug --recompute-features --eval-every 50 --num-epochs 3 --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline  

## Commands for Evaluation
Eval Example  
python main.py --do-eval --finetune-name synth_aug --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline  

Eval Non-finetuned Checkpoint  
python main.py --do-eval --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline 

Error Analysis (output predictions with ground truth, question, context, F1, and EM scores)
python main.py --do-eval --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline --error-file error_analysis.csv

Ensemble Evaluation Example
python main.py --do-eval --finetune-name aug1,aug1,aug1,synth_aug1 --eval-dir datasets/oodomain_val --save-dir save/adv_wiki_synth_aug1,save/adv_wiki_ood,save/fancyAdv3_multi_synth_aug1,save/adv_wiki_ind
