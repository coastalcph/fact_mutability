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


### Inference
This code passes a set of queries (the validation split of TempLama by default) through a language model and stores and model's hidden states and softmax scores in `outputs.json` and the model's predictions in `predictions.json`.
```
python inference.py -h
```

