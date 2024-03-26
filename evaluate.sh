# # alpaca
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_0---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_1---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_2---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_3---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/alpaca_fmv2_final_4---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name alpaca_4

# # lama
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_0---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_1---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_2---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_3---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama_fmv2_final_4---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_4

# lama2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2_0--meta-llama-Llama-2-7b-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama2_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2_1--meta-llama-Llama-2-7b-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama2_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2_2--meta-llama-Llama-2-7b-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama2_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2_3--meta-llama-Llama-2-7b-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama2_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2_4--meta-llama-Llama-2-7b-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama2_4

# # lama chat
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2-chat-7b_0--meta-llama-Llama-2-7b-chat-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_chat_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2-chat-7b_1--meta-llama-Llama-2-7b-chat-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_chat_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2-chat-7b_2--meta-llama-Llama-2-7b-chat-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_chat_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2-chat-7b_3--meta-llama-Llama-2-7b-chat-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_chat_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/llama2-chat-7b_4--meta-llama-Llama-2-7b-chat-hf/predictions.json --aliases_path coastalcph/fm_aliases --exp_name llama_chat_4

# falcon
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-7b_no_instr_0--tiiuae-falcon-7b/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-7b_no_instr_1--tiiuae-falcon-7b/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-7b_no_instr_2--tiiuae-falcon-7b/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-7b_no_instr_3--tiiuae-falcon-7b/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-7b_no_instr_4--tiiuae-falcon-7b/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_4

# falcon instruct
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-instruct-7b_0--tiiuae-falcon-7b-instruct/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_instruct_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-instruct-7b_1--tiiuae-falcon-7b-instruct/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_instruct_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-instruct-7b_2--tiiuae-falcon-7b-instruct/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_instruct_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-instruct-7b_3--tiiuae-falcon-7b-instruct/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_instruct_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/falcon-instruct-7b_4--tiiuae-falcon-7b-instruct/predictions.json --aliases_path coastalcph/fm_aliases --exp_name falcon_instruct_4

# flant5
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/flant5-xxl_0--google-flan-t5-xxl/predictions.json --aliases_path coastalcph/fm_aliases --exp_name flant5_0
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/flant5-xxl_1--google-flan-t5-xxl/predictions.json --aliases_path coastalcph/fm_aliases --exp_name flant5_1
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/flant5-xxl_2--google-flan-t5-xxl/predictions.json --aliases_path coastalcph/fm_aliases --exp_name flant5_2
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/flant5-xxl_3--google-flan-t5-xxl/predictions.json --aliases_path coastalcph/fm_aliases --exp_name flant5_3
WANDB_MODE="offline" python -m evaluation --data_path coastalcph/fm_queries --predictions_path ../fm_predictions/fm_queries_v2/flant5-xxl_4--google-flan-t5-xxl/predictions.json --aliases_path coastalcph/fm_aliases --exp_name flant5_4