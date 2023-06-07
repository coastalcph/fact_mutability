import argparse
import json
from tqdm import tqdm
import os
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_BEAMS = 10
MAX_ANSWER_LENGTH=10

TEMPLATES={'query_in_instructions': (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}: {}\n\n### Response:"
        ),
        'query_in_response': (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:{}"
        ),
        'query_in_input': (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
        )
        }

def prepare_prompt(query, args):
    if 'alpaca' in args.model_name_or_path:
        instruction = args.instruction 
        template = TEMPLATES[args.template]
        return template.format(instruction, query)
    else:
        return query
    
def get_scores(sequences, model, input_ids, prompt, query, tokenizer):
    
    logits = model(sequences)['logits']
    # we only case about the scores of the actual answer so we have to trim the  sequences
    trimmed_sequences = []
    trimmed_logits = []
    offset = input_ids.shape[-1] # omit outputs for the prompt
    for sequence, l in zip(sequences, logits):
        # alpaca models tend to repeat the query - we want to omit that
        if tokenizer.decode(sequence[offset:], skip_special_tokens=True).startswith(query):
            offset = len(tokenizer.encode(prompt + query, add_special_tokens=True))
        # at the front (dropping the prompt and a possibly repeated query)
        sequence = sequence[offset:].cpu().tolist()
        # at the back (dropping padding and punctuation)
        sequence = [idx for idx in sequence if idx not in [0,1,2,29889]] 
        # to max_length
        sequence = sequence[:MAX_ANSWER_LENGTH]
        trimmed_sequences.append(sequence)
        trimmed_logits.append(l[offset - 1: offset - 1 + len(sequence)])

    distributions = [torch.softmax(l, 1) for l in trimmed_logits]
    token_scores = [[distribution[i][sequence[i]].item() for i in range(len(sequence))] 
                                              for distribution, sequence in zip(distributions, trimmed_sequences)]
    overall_scores = [np.mean(np.log(ts)) for ts in token_scores]

    return trimmed_sequences, token_scores, overall_scores
            
def inference(dataset, tokenizer, model, args):
    config = GenerationConfig(
            max_new_tokens=50,
            num_beams=NUM_BEAMS,
            do_sample=False,
            output_hidden_states=False,
            output_scores=False,
            num_return_sequences=NUM_BEAMS,
            return_dict_in_generate=True
            )

    predictions = []
    outputs = {key: [] for key in ['raw_predictions', 'predictions']}
    for line in tqdm(dataset):
        qcode, query = line.split('\t')
        with torch.no_grad():
            prompt = prepare_prompt(query, args)
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            model_output = model.generate(input_ids, generation_config=config)
        
        answers, token_scores, overall_scores = get_scores(model_output['sequences'], model, input_ids, prompt, query, tokenizer)
        outputs['raw_predictions'].append({"qcode": qcode, "query": query, "predictions": [{'output_ids': model_output['sequences'][i].cpu().tolist(),
                                                                                            'answer': tokenizer.decode(model_output['sequences'][i])}
                                                                                                                          for i in range(NUM_BEAMS)]})
        outputs['predictions'].append({"qcode": qcode, "query": query, "predictions": [{'answer': tokenizer.decode(answers[i]),
                                                                                              'token_scores': token_scores[i],
                                                              'overall_score': overall_scores[i]} for i in range(NUM_BEAMS)]})
    return outputs

def main(args):

    print('Loading model')
    if 'alpaca' in args.model_name_or_path:
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()
    
    print('Loading dataset')
    dataset = open(args.queries_path).read().strip().split('\n')

    print('Running inference')
    outputs = inference(dataset, tokenizer, model, args)
    
    print('Writing outputs')
    experiment_name = '{}--{}'.format(args.exp_name, args.model_name_or_path.replace('/', '-'))
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    for key in outputs:
        with open(os.path.join(experiment_dir, key + '.json'), 'w') as outfile:
            for i, item in enumerate(outputs[key]):
                outfile.write(json.dumps(item))
                if i != len(outputs[key]) - 1:
                    outfile.write('\n')
    with open(os.path.join(experiment_dir,'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--queries_path", type=str, default='data/val.txt', help="Path to txt file, one query per line")
    parser.add_argument("--template", type=str, default='query_in_instructions', help="query_in_instructions, query_in_response or query_in_input")
    parser.add_argument("--instruction", type=str, default="Complete the fact in as few words as possible")
    parser.add_argument("--output_dir", type=str, default='output', help="Dir where model outputs will be stored")
    parser.add_argument("--exp_name", type=str, default='debug', help="Experiment name")
    parser.add_argument("--model_name_or_path", type=str, default="huggyllama/llama-7b", help="Model name or path")
    args = parser.parse_args()
 
    main(args)
