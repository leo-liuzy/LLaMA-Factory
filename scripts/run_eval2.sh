export CUDA_VISIBLE_DEVICES=2

python scripts/query_vllm.py --model-name-or-path saves/pt_on_ctrl_re_ood_both_lr1e-5 --context-type no-context --test-set-choice test_ood-both

python scripts/query_vllm.py --model-name-or-path saves/pt_on_ctrl_re_id_lr1e-5 --context-type no-context  --test-set-choice test_id
