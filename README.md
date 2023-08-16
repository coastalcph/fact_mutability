# fact_mutability

### Setup

```
conda create -n fact_mutability
conda activate fact_mutability
pip install torch==2.0.0
pip install git+https://github.com/huggingface/transformers
conda install matplotlib

# (Optional) For the iPython kernel
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=fact_mutability
```

Download and prepare the data
```
./get_data.sh
```

### Split queries in never, rarely, and often mutable

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


### TempLAMA/Validation Results

#### Without aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 12.8 |
| huggyllama/llama-7b     | first token     | 12.3 |
| chavinlo/alpaca-native  | perplexity      | 17.7 |
| chavinlo/alpaca-native  | first token     | 15.6 |
| google/flan-t5-xxl      | either          | 7.7  |

#### With aliases (target_mode: most recent answer)

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 14.9 |
| huggyllama/llama-7b     | first token     | 14.4 |
| chavinlo/alpaca-native  | perplexity      | 20.7 |
| chavinlo/alpaca-native  | first token     | 18.8 |
| google/flan-t5-xxl      | either          | 10.1 |

### LAMA Results

#### Without aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| google/flan-t5-xxl      | either          | 34.1 |

#### With aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| google/flan-t5-xxl      | either          | 40.9 |

### Immutable Results

#### Without aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 27.7 |
| huggyllama/llama-7b     | first token     | 27.7 |
| chavinlo/alpaca-native  | perplexity      | 54.1 |
| chavinlo/alpaca-native  | first token     | 34.4 |
| google/flan-t5-xxl      | either          | 53.2 |

#### With aliases 

| Model                   | Prediction mode |  F1  |
| ----------------------  | --------------- | ---- |
| huggyllama/llama-7b     | perplexity      | 30.8 |
| huggyllama/llama-7b     | first token     | 30.7 |
| chavinlo/alpaca-native  | perplexity      | 57.2 |
| chavinlo/alpaca-native  | first token     | 38.0 |
| google/flan-t5-xxl      | either          | 56.6 |
