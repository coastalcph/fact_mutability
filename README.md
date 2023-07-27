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

### Predictions
[Here](https://huggingface.co/spaces/Yova/fm_predictions) (`qir` means query in response)


### Validation Results

#### Without aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 12.8 |
| huggyllama/llama-7b     | first token     | 12.3 |
| chavinlo/alpaca-native  | perplexity      | 17.7 |
| chavinlo/alpaca-native  | first token     | 15.6 |

#### With aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 14.9 |
| huggyllama/llama-7b     | first token     | 14.4 |
| chavinlo/alpaca-native  | perplexity      | 20.7 |
| chavinlo/alpaca-native  | first token     | 18.8 |

### LAMA Results

#### Without aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 35.5 |
| huggyllama/llama-7b     | first token     | 30.8 |
| chavinlo/alpaca-native  | perplexity      | 69.7 |
| chavinlo/alpaca-native  | first token     | 33.4 |

#### With aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 41.5 |
| huggyllama/llama-7b     | first token     | 36.3 |
| chavinlo/alpaca-native  | perplexity      | 73.5 |
| chavinlo/alpaca-native  | first token     | 39.0 |

### Immutable Results

#### Without aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 27.7 |
| huggyllama/llama-7b     | first token     | 27.7 |
| chavinlo/alpaca-native  | perplexity      | 54.1 |
| chavinlo/alpaca-native  | first token     | 34.4 |

#### With aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 30.8 |
| huggyllama/llama-7b     | first token     | 30.7 |
| chavinlo/alpaca-native  | perplexity      | 57.2 |
| chavinlo/alpaca-native  | first token     | 38.0 |
