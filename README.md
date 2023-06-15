# fact_mutability

### Setup

```
conda create -n fact_mutability
conda activate fact_mutability
pip install torch==2.0.0
pip install git+https://github.com/huggingface/transformers
```
Download and prepare the data
```
./get_data.sh
```

### Split queries in never, rarely, and oftern mutable

```
python -m utils.split_mutable
```


### Inference
This code passes a set of queries (one query per line) through a language model and stores the model's predictions and softmax scores in `predictions.json`. If model name contains the string "alpaca" the query is placed inside a template (three options) with a user-specified instruction.
Uses beam search (10) to generate predictions and stores all. 
```
python inference.py -h
```

### Evaluation
SQUAD-style F1-score evaluation, where the user specifies whether to select the best prediction based on perplexity of first token score (`prediction_mode`)  and whether to evaluate against the most recent, most frequent, or year-specific answer (`target_mode`). If you pass `data/<SPLIT>.json` as `data_path`, it evaluates on the original templama answer only, and if you pass `data/<SPLIT>_with_aliases.json`, it evaluates with all aliases for the correct answer.
```
python evaluation.py --predictions_path <PATH>
```
### Validation Results

#### Without aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      |  |
| huggyllama/llama-7b     | first token     |  |
| chavinlo/alpaca-native  | perplexity      | 17.7 |
| chavinlo/alpaca-native  | first token     | 15.6 |

#### With aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      |  |
| huggyllama/llama-7b     | first token     |  |
| chavinlo/alpaca-native  | perplexity      | 20.7 |
| chavinlo/alpaca-native  | first token     | 18.8 |
