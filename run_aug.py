# import dataclasses
import json
# import logging
# import os
# import sys
# from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import random
import hashlib

# from transformers import (
#     AutoModelForSeq2SeqLM,
#     AutoTokenizer,
#     T5Tokenizer,
#     BartTokenizer,
#     HfArgumentParser,
#     DataCollator,
#     TrainingArguments,
#     set_seed,
# )

# from question_generation.qg_trainer import Trainer
# from question_generation.data_collator import T2TDataCollator
# from question_generation.qg_utils import freeze_embeds, assert_not_all_frozen
from util import read_squad, write_squad

from question_generation.pipelines import pipeline
from nltk import sent_tokenize
import json

import argparse
import re

import nlpaug.augmenter.word as naw

# MODEL_TYPE_TO_TOKENIZER = {
#     "t5": T5Tokenizer,
#     "bart": BartTokenizer,
# }


# logger = logging.getLogger(__name__)

# Question Answer Pair Generation
# This function generates synthetic question answer pairs
# given a context paragraph. It uses the original squad 
# trained model from the paper https://arxiv.org/pdf/1906.05416v1.pdf
# Question answer pairs are generated per sentence (given snapshot of larger
# context window). Context windows are 16 sentences at a time (limitation of 
# the model). We also use roundtrip consistency to validate the
# generated answers match a QA model's prediction.
def gen_qas(synth_file):
    dataset_dict = read_squad(synth_file)
    context_list = list(set(dataset_dict['context']))
    
    json_output = {'data':[]}

    chunked_context = {'full_context': [], 'context': []}

    # Chunk the context since some context are too long
    for full_context in context_list:
        # split into 16 sentences sliding 8 at a time
        sents = sent_tokenize(full_context)

        stride = 16
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

    # We use the original squad trained model from the paper https://arxiv.org/pdf/1906.05416v1.pdf
    nlp = pipeline("multitask-qa-qg")
    num_questions = 0
    num_mismatch = 0

    # generate qa pairs for each context
    total_chunks = len(chunked_context['context'])
    # used to remove duplicate questions
    total_questions = set()

    # HACK parameters when generating synthetic for indomain
    # since takes too long
    # random_idx = random.sample(range(len(chunked_context['context'])), 5000)
    # for idx in random_idx:

    # Process each chunked context
    for idx, context in enumerate(chunked_context['context']):
        context = chunked_context['context'][idx]

        # HACK parameters when generating synthetic for indomain
        # since takes too long
        # Does a partial sample for generating questions
        # if random.random() >= 0.25:
            # continue

        # return qa pairs
        qas = nlp(context)
        if len(qas) == 0:
            continue

        # roundtrip consistency
        # run qa model to see if it returns same as qa pair
        # only keep qas that qa model also predicts
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

        # HACK parameters when generating synthetic for indomain
        # since takes too long
        # limit to 5 questions
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

    ## Write out synthetic examples
    with open(synth_file + '_synth', 'w') as outfile:
        json.dump(json_output, outfile)
        print(f'Synthetic Examples written to {synth_file}_synth')
    
    # combine_qas(synth_file)

# Data Augmentation
# Perform random synonym replacement for our question answer context tuples
# Highlight the answer in the context first
# Then run augmentation
# Then extract answer again so if answer span is replaced with synonym,
# both the answer and context are updated properly.
# We use the nlpaug package for the synonym replacement step.
def data_aug(file):
    dataset_dict = read_squad(file)

    hl_contexts = []
    hl_questions = []

    # highlight the context
    for i in range(len(dataset_dict['context'])):
        context = dataset_dict['context'][i]
        question = dataset_dict['question'][i]
        answer = dataset_dict['answer'][i]
        answer_text = answer['text'][0]
        ans_start_idx = answer['answer_start'][0]

        hl_contexts.append(f'{context[:ans_start_idx]} <hl> {answer_text} <hl> {context[ans_start_idx + len(answer_text): ]}')
        hl_questions.append(question)

    aug_samples = {'question': [], 'context': [], 'id': [], 'answer': []}

    # augment with wordnet synonym
    aug = naw.SynonymAug(aug_src='wordnet', stopwords=['<hl>', '\n', '\t', "\'"], aug_min=1, aug_max=5)

    for idx, context in enumerate(hl_contexts):
        print(f'{idx} of {len(hl_contexts)}')
        # generate augmentation multiple times per answer 
        for rpt in range(5):
            aug_context = ''
            # ensure 2 <hl> exist still
            while aug_context.count('<hl>') != 2:
                # aug = naw.WordEmbsAug(
                #     model_type='word2vec', model_path= 'nlpaug/model/GoogleNews-vectors-negative300.bin',
                #     # stopwords=['<hl>', '\n', '\t', "\'"], aug_min=5, aug_max=20,
                #     action="substitute")
                aug_context = aug.augment(context)
                aug_context = aug_context.replace(" \' ", "\'")
                aug_context = aug_context.replace("<hl >", "<hl>")
                aug_context = aug_context.replace("< hl>", "<hl>")
                aug_context = aug_context.replace("< hl >", "<hl>")
                aug_context = aug_context.replace(" )", ")")
                aug_context = aug_context.replace(" .", ".")
                aug_context = aug_context.replace(" - ", "-")
                
            # extract answer text and index
            re_span = re.search('<hl>(.+?)<hl>', aug_context)
            answer_text = re_span.group(1).strip()
            answer_start_idx = re_span.start()

            answer = {"answer_start": [answer_start_idx], "text": [answer_text]}

            # store in dict
            # remove <hl> from context
            cleaned_context = aug_context.replace('<hl>', '')
        
            aug_samples['question'].append(hl_questions[idx])
            aug_samples['id'].append(hashlib.md5((question + answer_text).encode('utf-8')).hexdigest())
            aug_samples['context'].append(cleaned_context)
            aug_samples['answer'].append(answer)

    write_squad(aug_samples, file + '_aug')
    
# Combine Variants
# This function combines multiple files into one file
# It allows us to merge synthetic and augmented data into 1 file
# Shared contexts are merged so overall file can be smaller
def combine_qas(filepath, variants):
    dataset_dict = read_squad(filepath)

    for variant in variants:
        variant_dict = read_squad(filepath + '_' + variant)

        dataset_dict['question'].extend(variant_dict['question'])
        dataset_dict['context'].extend(variant_dict['context'])
        dataset_dict['id'].extend(variant_dict['id'])
        dataset_dict['answer'].extend(variant_dict['answer'])

    write_squad(dataset_dict, filepath + '_combined')

def get_action_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth-file', type=str, default='')
    parser.add_argument('--combine', type=str, default='')
    parser.add_argument('--aug', type=str, default='')
    parser.add_argument('--variants', type=str, default='synth')
    args = parser.parse_args()
    return args

def main(args_file=None):
    action_args = get_action_args()
    
    ## Generate Synthetic QA Pairs
    if action_args.synth_file != '':
        print(f'Generating synthetic question answer pairs for {action_args.synth_file}')
        gen_qas(action_args.synth_file)
        return
    ## Combine Variants
    if action_args.combine != '':
        print(f'Combining {action_args.combine} with {action_args.variants} data')
        combine_qas(action_args.combine, action_args.variants.split(','))
        return
    ## Data Aug
    if action_args.aug != '':
        print(f'Generating augmented data for {action_args.aug}')
        data_aug(action_args.aug)
        return

if __name__ == "__main__":
    main()