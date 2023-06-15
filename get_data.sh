mkdir data
cd data

# original
wget https://storage.googleapis.com/gresearch/templama/train.json
wget https://storage.googleapis.com/gresearch/templama/val.json
wget https://storage.googleapis.com/gresearch/templama/test.json

# with aliases 
wget https://huggingface.co/datasets/Yova/templama/resolve/main/train_with_aliases.json
wget https://huggingface.co/datasets/Yova/templama/resolve/main/val_with_aliases.json
wget https://huggingface.co/datasets/Yova/templama/resolve/main/test_with_aliases.json

# immutable
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/immutable.json
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/immutable_with_aliases.json

cd ..
python utils/extract_queries_only.py
