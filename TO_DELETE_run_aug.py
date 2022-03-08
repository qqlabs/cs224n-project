from util import read_squad, write_squad
import argparse
import json
import re
import hashlib

import nlpaug.augmenter.word as naw

def data_aug(synth_file):
    dataset_dict = read_squad(synth_file)

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
    # augment
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

    write_squad(aug_samples, synth_file + '_aug')
    

def get_action_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='')
    args = parser.parse_args()
    return args

def main(args_file=None):
    action_args = get_action_args()
    
    ## Generate Synthetic QA Pairs
    if action_args.file != '':
        print(f'Generating augmented data for {action_args.file}')
        data_aug(action_args.file)
        return

if __name__ == "__main__":
    main()