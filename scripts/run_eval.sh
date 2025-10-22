export CUDA_VISIBLE_DEVICES=1

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type no-context --test-set-choice test_id

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type no-context --test-set-choice test_ood-both

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type gold-context --test-set-choice test_id

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type gold-context --test-set-choice test_ood-both

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type rag-context --test-set-choice test_id

python scripts/query_vllm.py --model-name-or-path /u/zliu/datastor1/shared_resources/models/qwen/Qwen3-1.7B-Base --context-type rag-context --test-set-choice test_ood-both