import argparse
import json
from tqdm import tqdm
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(dataset, tokenizer, model):
    config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True
            )

    predictions = []
    outputs = []
    for query in tqdm(dataset):
        with torch.no_grad():
            input_ids = tokenizer.encode(query, return_tensors='pt').to(device)
            output = model.generate(input_ids, generation_config=config)
        
        prediction = tokenizer.decode(output['sequences'][0], skip_special_tokens=True)
        predictions.append({'query': query, 'prediction': prediction.replace(query, '')})
        
        output['sequences'] = output['sequences'].cpu().tolist()
        output['scores'] = [x.cpu().tolist() for x in output['scores']]
        output['hidden_states'] = [[x.cpu().tolist() for x in item] for item in output['hidden_states']]
        outputs.append(output)

    return predictions, outputs

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    dataset = open(args.queries_path).read().strip().split('\n')

    predictions, outputs = inference(dataset, tokenizer, model)
    
    experiment_name = '{}--{}'.format(args.exp_name, args.model_name_or_path.replace('/', '-'))
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.mkdir(experiment_dir)
    with open(os.path.join(experiment_dir, 'predictions.json'), 'w') as outfile:
        json.dump(predictions, outfile)
    with open(os.path.join(experiment_dir, 'outputs.json'), 'w') as outfile:
        json.dump(outputs, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--queries_path", type=str, default='data/val.txt', help="Path to txt file, one query per line")
    parser.add_argument("--output_dir", type=str, default='output', help="Dir where model outputs will be stored")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--model_name_or_path", type=str, default="huggyllama/llama-7b", help="Model name or path")
    args = parser.parse_args()
 
    main(args)
