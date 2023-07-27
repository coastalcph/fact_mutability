mkdir data
cd data

# original
mkdir templama
cd templama
wget https://storage.googleapis.com/gresearch/templama/train.json
wget https://storage.googleapis.com/gresearch/templama/val.json
wget https://storage.googleapis.com/gresearch/templama/test.json

# with aliases 
wget https://huggingface.co/datasets/Yova/templama/resolve/main/train_with_aliases.json
wget https://huggingface.co/datasets/Yova/templama/resolve/main/val_with_aliases.json
wget https://huggingface.co/datasets/Yova/templama/resolve/main/test_with_aliases.json
cd ..

# immutable
mkdir immutable
cd immutable
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/immutable/data.json
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/immutable/data_with_aliases.json
cd ..

# lama
mkdir lama
cd lama
## STEPS FOR CREATING THE DATA
#wget https://dl.fbaipublicfiles.com/LAMA/data.zip
#unzip data.zip
#mv data/TREx .
#mv data/relations.jsonl .
#rm data.zip
#rm -r data
#cd ..
##
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/lama/data.json
wget https://huggingface.co/datasets/Yova/immutable_facts/resolve/main/lama/data_with_aliases.json
cd..

cd ..
#python utils/format_lama_data.py
#rm -r data/lama/TREx
python utils/extract_queries_only.py
