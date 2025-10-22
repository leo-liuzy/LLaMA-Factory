import os
import json
import argparse
import requests
from tqdm import tqdm
# from reason import util
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, RequestOutput

import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer

from knowledge_propagation.utils import io
from knowledge_propagation.modules.dual_retriever import DualRetrieverLite


from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd
from glob import glob
from bespokelabs import curator
from datasets import Dataset

score_tag_extractor = extractor.tag_content_extractor("score")

ICL_examples = """
Question: Where in England was Dame Judi Dench born?
Answer: <answer>York</answer>
Question: From which country did Angola achieve independence in 1975?
Answer: <answer>Portugal</answer>
Question: Which city does David Soul come from?
Answer: <answer>Chicago</answer>
Question: Who won Super Bowl XX?
Answer: <answer>Chicago Bears</answer>
Question: Which was the first European country to abolish capital punishment?
Answer: <answer>Norway</answer>
Question: In which country did he widespread use of ISDN begin in 1988?
Answer: <answer>Japan</answer>
Question: What is Bruce Willis' real first name?
Answer: <answer>Walter</answer>
Question: Which William wrote the novel Lord Of The Flies?
Answer: <answer>Golding</answer>
Question: How is Joan Molinsky better known?
Answer: <answer>Joan Rivers</answer>
Question: In which branch of the arts is Patricia Neary famous?
Answer: <answer>Ballet</answer>
""".strip()

class LlmAsJudge(curator.LLM):
    MAX_VAL: float = 10.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 0: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        score /= self.MAX_VAL
        input["llm_accuracy-soft"] = score

        return {**input}

class LlmAsJudgeHard(curator.LLM):
    MAX_VAL: float = 1.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
accuracy: 
                Score 1: The answer is completely accurate and aligns perfectly with the reference.
                Score 0: Otherwise.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Ground truth]
{reference}
[The End of Ground truth]

[The Start of Assistant's Answer]
{prediction}
[The End of Assistant's Answer]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"], reference=input["answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        score /= self.MAX_VAL
        input["llm_accuracy-hard"] = score

        return {**input}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query vLLM with MATH-500 examples"
    )
    parser.add_argument(
        "--model-name-or-path", type=str, required=True,
        help="Model name to query (should match the model served by vLLM)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=1,
        help="Top-p sampling parameter (nucleus sampling)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=30,
        help="Number of samples to generate per example"
    )
    parser.add_argument(
        "--eval-data-name", type=str, default="triviaqa", choices=["all", "ctrl_RE", "triviaqa"],
        help="Dataset name"
    )
    parser.add_argument(
        "--triviaqa_version", type=str, default="rc.nocontext",
        help="TriviaQA configuration: rc.nocontext | rc | unfiltered.nocontext"
    )
    parser.add_argument(
        "--triviaqa_split", type=str, default="validation",
        help="TriviaQA split: train | validation | test"
    )
    parser.add_argument(
        "--test-set-choice", type=str, default="compositional_celebrities_reformat",
        help="Test set choice"
    )
    parser.add_argument(
        "--llm-judge-name", type=str, default="gpt-4o-mini",
        help="LLM judge type",
    )

    parser.add_argument(
        "--context-type", type=str, default="no-context", choices=["no-context", "gold-context", "rag-context", "self-ask", "self-ask_system", "self-ask_musique", "self-ask_system_musique", "self-ask_w-instruction", "self-ask_w-instruction_musique", "self-ask_w-instruction_system", "self-ask_w-instruction_system_musique"],
        help="Context type",
    )

    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite the existing results"
    )
    parser.add_argument(
        "--question_key", type=str, default="questions",
        help="Question key"
    )
    return parser.parse_args()



def load_controlled_RE_data(args, file_path):
    samples = io.load_jsonlines(file_path)[:]
    return Dataset.from_list(samples)


def get_messages_from_problem(problem, model_name_or_path_base, dataset_name="ctrl_RE", contexts=None, context_type="no-context"):
    """Extract messages from problem for vLLM API"""

    if dataset_name in ["ctrl_RE", "triviaqa"]:
        # return [
        #     {"role": "user", "content": problem}
        # ]
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Provides a *short* final answer. The *short* final answer is enclosed between <answer> </answer> tags." + "\n" + ICL_examples
        
        if context_type == "no-context":
            context = "Question: "
        else:
            raise ValueError(f"Invalid context type: {context_type}")
        # 
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context + problem}
        ]   
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def main():
    args = parse_args()
    
    if args.eval_data_name == "all":
        eval_data_names = ["ctrl_RE", "triviaqa"]
    else:
        eval_data_names = [args.eval_data_name]
    model = None
    
    for eval_data_name in eval_data_names:
        model_name_or_path_base = os.path.basename(args.model_name_or_path)
        save_dir = f"/u/zliu/datastor1/LLaMA-Factory/eval_saves/{model_name_or_path_base}"
        os.makedirs(save_dir, exist_ok=True)
        if eval_data_name == "triviaqa":
            output_file = f"{save_dir}/{eval_data_name}_{args.triviaqa_version}_{args.triviaqa_split}.jsonl"
        else:
            output_file = f"{save_dir}/{eval_data_name}_{args.test_set_choice}_{args.question_key}_{args.context_type}.jsonl"
    
        if not os.path.exists(output_file) or args.overwrite:
            
            sampling_params = SamplingParams(
                max_tokens=args.max_tokens, 
                top_p=args.top_p,
                temperature=args.temperature,
                n=args.num_samples,
                skip_special_tokens=False,
            )
            # import pdb; pdb.set_trace()
            if eval_data_name == "ctrl_RE":
                dataset = load_controlled_RE_data(args, f"/u/zliu/datastor1/LLaMA-Factory/data/{args.test_set_choice}.jsonl")
                problem_key = "question"
                answer_key = "answer"
            elif eval_data_name == "triviaqa":
                dataset = load_dataset("trivia_qa", args.triviaqa_version, split=args.triviaqa_split)
                problem_key = "question"
                answer_key = "answer"  # dict with value/aliases
            else:
                raise ValueError(f"Invalid dataset name: {eval_data_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            if args.context_type == "no-context":
                if eval_data_name == "ctrl_RE":
                    all_messages = [get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name, contexts=None, context_type=args.context_type) for d in dataset]
                else:
                    all_messages = [
                        get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name, contexts=None, context_type=args.context_type)
                        for d in dataset
                    ]
            else:
                raise ValueError(f"Invalid context type: {args.context_type}")
            
            # import pdb; pdb.set_trace()
            
            try:
                texts = tokenizer.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                # texts = [m[-1]["content"] for m in all_messages]
                
                if model is None:
                    model = LLM(args.model_name_or_path)
                outputs = model.generate(texts[:], sampling_params=sampling_params)
                results = []
                
                for idx, output in enumerate(outputs):
                    
                    for sample_idx, out in enumerate(output.outputs):
                        
                        model_answer = out.text
                        # import pdb; pdb.set_trace()
                        model_answer = model_answer.split("</think>")[-1].strip()
                        # Ground truth answer formatting
                        if eval_data_name == "triviaqa":
                            gt = dataset[idx][answer_key]
                            if isinstance(gt, dict):
                                if isinstance(gt.get("normalized_aliases"), list) and len(gt["normalized_aliases"]) > 0:
                                    ground_truth = "; ".join(gt["normalized_aliases"])
                                elif isinstance(gt.get("aliases"), list) and len(gt["aliases"]) > 0:
                                    ground_truth = "; ".join(gt["aliases"])
                                else:
                                    ground_truth = str(gt.get("value", ""))
                            else:
                                ground_truth = str(gt)
                            qtype = dataset[idx].get("question_id", "triviaqa")
                        else:
                            ground_truth = dataset[idx][answer_key]
                            qtype = dataset[idx]["type"]

                        results.append({
                            "prompt": texts[idx],
                            "question": dataset[idx][problem_key],
                            "eval_data_name": eval_data_name,
                            "ground_truth_answer": ground_truth,
                            "sample_id": sample_idx,
                            "model_answer": model_answer,
                            "model_response": out.text,
                            "question_type": qtype,
                        })
            except Exception as e:
                print(f"\nError processing: {e}")
            # import pdb; pdb.set_trace()
            # df = pd.DataFrame(results)
            # df.to_excel(output_file, index=False)
            io.dump_jsonlines(results, output_file)
            # print(f"\nFinal results saved to {output_file}")
        
        # evaluate with llm_judge
        
        # for llm_judge_type in ["hard",]:
        #     df = pd.read_excel(output_file)
        #     if f"llm_accuracy-{llm_judge_type}" not in df.columns or args.overwrite:
        #         print(f"Evaluating with [{llm_judge_type}] judge")
        #         llm_judge_name = args.llm_judge_name
        #         if llm_judge_type == "hard":
        #             llm_judge = LlmAsJudgeHard(
        #                 model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
        #             )
        #         else:
        #             llm_judge = LlmAsJudge(
        #                 model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
        #             )
        #         df["predicted_answer"] = df["model_response"].astype(str)
        #         df["answer"] = df["ground_truth_answer"].astype(str)
        #         scored_dataset = Dataset.from_pandas(df[:])
        #         scored_dataset = llm_judge(
        #             scored_dataset,
        #         )
        #         # import pdb; pdb.set_trace()
        #         ds = scored_dataset
        #         if hasattr(scored_dataset, "dataset"):
        #             ds = scored_dataset.dataset
                    
        #         scored_df = ds.to_pandas().drop(columns=['predicted_answer', 'answer',])
        #         scored_df["llm_judge"] = llm_judge_name
        #         scored_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    main()