import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import random

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    set_seed,
)

from question_generation.qg_trainer import Trainer
from question_generation.data_collator import T2TDataCollator
from question_generation.qg_utils import freeze_embeds, assert_not_all_frozen
from util import read_squad

from question_generation.pipelines import pipeline
from nltk import sent_tokenize
import json

import argparse

MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={"help": "label smoothing rate, set to > 0 if you want to enable lable smoothing"}
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"}, 
    )
    task: Optional[str] = field(
        default=None,
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    qg_format: Optional[str] = field(
        default='prepend_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

def gen_qas(synth_file):
    dataset_dict = read_squad(synth_file)
    context_list = list(set(dataset_dict['context']))
    
    json_output = {'data':[]}

    chunked_context = {'full_context': [], 'context': []}

    for full_context in context_list:
        # split into 16 sentences sliding 8 at a time
        sents = sent_tokenize(full_context)

        stride = 8
        max_len = 16
        i = 0
        chunk_end = 0

        while chunk_end < len(sents):
            chunk_start = i*stride
            chunk_end = i*stride + max_len
            sent_chunk = " ".join(sents[chunk_start:chunk_end])
            chunked_context['full_context'].append(full_context)
            chunked_context['context'].append(sent_chunk)
            i += 1

    nlp = pipeline("multitask-qa-qg")
    num_questions = 0
    num_mismatch = 0

    # generate qa pairs for each context
    # TODO remove duplicate questions
    total_chunks = len(chunked_context['context'])

    total_questions = set()
    for idx, context in enumerate(chunked_context['context']):
        # if random.random() >= 0.01:
        #     continue
        # return qa pairs
        qas = nlp(context)
        if len(qas) == 0:
            continue

        # roundtrip consistency
        # run qa model to see if it returns same as qa pair
        qas_filtered = []
        for qa in qas:
            if qa['question'] in total_questions:
                continue
            total_questions.add(qa['question'])
            num_questions += 1
            qa_ans = nlp({'question': qa['question'],
                'context': context
            })
            if qa_ans != qa['answers'][0]['text']:
                num_mismatch += 1
                continue
            qas_filtered.append(qa)

        # limit to 2 questions
        # qas_filtered = random.sample(qas_filtered, min(2, len(qas_filtered)))

        full_context = chunked_context['full_context'][idx]
        title = full_context[:52]
        json_entry = {
            "title":title,
            "paragraphs":[{"context": full_context, "qas": qas_filtered}] 
        }
        json_output['data'].append(json_entry)

        print(f'{idx} out of {total_chunks}')
    
    print(f'Num Questions: {num_questions}')
    print(f'Num Mismatch: {num_mismatch}')

    ## TODO merge the context into one full_context json
    with open(synth_file + '_synth', 'w') as outfile:
        json.dump(json_output, outfile)
        print(f'Synthetic Examples written to {synth_file}_synth')
    
    combine_qas(synth_file)

def combine_qas(filepath):
    dataset_dict = read_squad(filepath)
    synth_dict = read_squad(filepath + '_synth')

    dataset_dict['question'].extend(synth_dict['question'])
    dataset_dict['context'].extend(synth_dict['context'])
    dataset_dict['id'].extend(synth_dict['id'])
    dataset_dict['answer'].extend(synth_dict['answer'])

    json_output = {'data':[]}

    sort_idx = sorted(range(len(dataset_dict['context'])), key=dataset_dict['context'].__getitem__)

    i = 0
    while i < len(sort_idx):
        full_context = dataset_dict['context'][sort_idx[i]]
        title = full_context[:52]
        qas = []
        while (i + 1 < len(sort_idx)):
            if (dataset_dict['context'][sort_idx[i+1]] != dataset_dict['context'][sort_idx[i]]):
                break
            # reiterate through answers to format output properly
            answers = []
            for a_idx in range(len(dataset_dict['answer'][sort_idx[i]]['answer_start'])):
                answers.append({'answer_start':dataset_dict['answer'][sort_idx[i]]['answer_start'][a_idx], 'text': dataset_dict['answer'][sort_idx[i]]['text'][a_idx]})
            qas.append({'question': dataset_dict['question'][sort_idx[i]], 'id': dataset_dict['id'][sort_idx[i]], 'answers': answers})
            i += 1
        answers = []
        for a_idx in range(len(dataset_dict['answer'][sort_idx[i]]['answer_start'])):
            answers.append({'answer_start':dataset_dict['answer'][sort_idx[i]]['answer_start'][a_idx], 'text': dataset_dict['answer'][sort_idx[i]]['text'][a_idx]})
        qas.append({'question': dataset_dict['question'][sort_idx[i]], 'id': dataset_dict['id'][sort_idx[i]], 'answers': answers})
        i += 1
        json_entry = {
            "title":title,
            "paragraphs":[{"context": full_context, "qas": qas}] 
        }
        json_output['data'].append(json_entry)

    with open(filepath + '_combined', 'w') as outfile:
        json.dump(json_output, outfile)
        print(f'Combined {filepath} with synth data.')

def get_action_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth-file', type=str, default='')
    parser.add_argument('--combine', type=str, default='')
    args = parser.parse_args()
    return args

def main(args_file=None):
    action_args = get_action_args()
    
    ## Generate Synthetic QA Pairs
    if action_args.synth_file != '':
        print(f'Generating synthetic question answer pairs for {action_args.synth_file}')
        gen_qas(action_args.synth_file)
        return
    if action_args.combine != '':
        print(f'Combining {action_args.combine} with synth data')
        combine_qas(action_args.combine)
        return

    ## QG Model Training 
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert model_args.model_type in list(MODEL_TYPE_TO_TOKENIZER.keys()), "model type should be 't5' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Set project name
    os.environ["WANDB_PROJECT"] = "question-generation"

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_embeds:
        logger.info("freezing embeddings of the model")
        freeze_embeds(model)
        assert_not_all_frozen(model)

    # Get datasets
    logger.info('loading dataset')
    
    train_dataset = torch.load(data_args.train_file_path) if training_args.do_train else None
    valid_dataset = torch.load(data_args.valid_file_path) if training_args.do_eval else None
    
    logger.info('finished loading dataset')

    # Initialize data_collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_args.model_type,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        prediction_loss_only=True,
        label_smoothing=model_args.label_smoothing
    )

    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def run_qg(args_dict):
    with open("args.json", 'w') as f:
        json.dump(args_dict, f)
    
    main(args_file="args.json")

if __name__ == "__main__":
    main()