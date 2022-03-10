# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

## Commands for Data Augmentation

Synthetic Examples
python run_aug.py --synth-file datasets/oodomain_train/duorc

Data Augmentation
python run_aug.py --repeat-aug 5 --aug datasets/oodomain_train/duorc

Combine Synthetic + Augmentation Variants
python run_aug.py --variants synth,aug --combine datasets/oodomain_train/duorc

## Commands for Training
* change training datasets with flags
  * ID: default
  * ID + OOD: --combined
  * ID + OOD augmented: --combinedwAug (need to change datasets
* use flag --adv-train for adversarial model
* use flag --wiki-align for wiki alignment

## Commands for Finetuning
* can finetune on OOD, OOD+synth, OOD+aug, or OOD+synth+aug
* use --variants flag to control what sample is being used for finetuning

Finetuning Example
python main.py --do-finetune --finetune-name synth_aug --variants synth,aug --recompute-features --eval-every 50 --num-epochs 3 --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline

## Commands for Evaluation
Eval Example
python main.py --do-eval --finetune-name synth_aug --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline

Eval Non-finetuned Checkpoint
python main.py --do-eval --eval-dir datasets/oodomain_val --save-dir save/adversarial-baseline
