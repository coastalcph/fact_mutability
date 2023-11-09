# alpaca
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_0---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_1---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_2---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_3---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_4---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_4

# lama
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_0---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_1---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_2---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_3---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_4---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_4