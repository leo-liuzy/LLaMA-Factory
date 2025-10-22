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

SELF_ASK_PROMPT = """
Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate answer: Harry Vaughan Watkins was 69 years old when he died.
So the final answer is: <answer>Harry Vaughan Watkins</answer> <confidence>0.9</confidence>

Question: Why did the founder of Versus die?
Are follow up questions needed here: Yes.
Follow up: Who founded Versus?
Intermediate answer: Gianni Versace.
Follow up: Why did Gianni Versace die?
Intermediate answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997.
So the final answer is: <answer>Shot</answer> <confidence>1.0</confidence>

Question: Who is the grandchild of Dambar Shah?
Are follow up questions needed here: Yes.
Follow up: Who is the child of Dambar Shah?
Intermediate answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.
Follow up: Who is the child of Krishna Shah?
Intermediate answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah.
So the final answer is: <answer>Rudra Shah</answer> <confidence>0.9</confidence>
""".strip()

SELF_ASK_PROMPT_MUSIQUE = """
Question: When does monsoon season end in the state the area code 575 is located?
Are follow up questions needed here: Yes.
Follow up: Which state is the area code 575 located in?
Intermediate answer: The area code 575 is located in New Mexico.
Follow up: When does monsoon season end in New Mexico?
Intermediate answer: Monsoon season in New Mexico typically ends in mid-September.
So the final answer is: <answer>mid-September</answer> <confidence>0.95</confidence>

Question: What is the current official currency in the country where Ineabelle Diaz is a citizen?
Are follow up questions needed here: Yes.
Follow up: Which country is Ineabelle Diaz a citizen of?
Intermediate answer: Ineabelle Diaz is from Peurto Rico, which is in the United States of America.
Follow up: What is the current official currency in the United States of America?
Intermediate answer: The current official currency in the United States is the United States dollar.
So the final answer is: <answer>United States dollar</answer> <confidence>0.9</confidence>

Question: Where was the person who founded the American Institute of Public Opinion in 1935 born?
Are follow up questions needed here: Yes.
Follow up: Who founded the American Institute of Public Opinion in 1935?
Intermediate answer: George Gallup.
Follow up: Where was George Gallup born?
Intermediate answer: George Gallup was born in Jefferson, Iowa.
So the final answer is: <answer>Jefferson</answer> <confidence>1.0</confidence>

Question: What language is used by the director of Tiffany Memorandum?
Are follow up questions needed here: Yes.
Follow up: Who directed the movie called Tiffany Memorandum?
Intermediate answer: Sergio Grieco.
Follow up: What language is used by Sergio Grieco?
Intermediate answer: Sergio Grieco speaks Italian.
So the final answer is: <answer>Italian</answer> <confidence>0.9</confidence>

Question: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
Are follow up questions needed here: Yes.
Follow up: Which player scored the first touchdown in Superbowl 1?
Intermediate answer: Max McGee.
Follow up: Which sports team did Max McGee play for?
Intermediate answer: Max McGee played for the Green Bay Packers.
So the final answer is: <answer>Green Bay Packers</answer> <confidence>0.9</confidence>

Question: The birth country of Jayantha Ketagoda left the British Empire when?
Are follow up questions needed here: Yes.
Follow up: What is the birth country of Jayantha Ketagoda?
Intermediate answer: Sri Lanka.
Follow up: When did Sri Lanka leave the British Empire?
Intermediate answer: Sri Lanka left the British Empire on February 4, 1948.
So the final answer is: <answer>February 4, 1948</answer> <confidence>1.0</confidence>
"""


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

class LlmAsJudgeAbstention(curator.LLM):
    MAX_VAL: float = 1.0
    PROMPT: str = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
                Score 1: The assistant is refusing to answer the question. e.g. "I'm sorry, I don't know."
                Score 0: Otherwise.
                Only respond with a numerical score.

[Question]
{question}

[The Start of Assistant's Response]
{prediction}
[The End of Assistant's Response]

Return the numerical score wrapped in <score>..</score> tag
    """.strip()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(
            question=input["question"], prediction=input["predicted_answer"]
        )

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        score_ = score_tag_extractor(response)
        assert len(score_) == 1
        score = score_[0].strip()
        assert score.isdigit()
        assert 0 <= float(score) <= self.MAX_VAL
        score = float(score)
        input["abstention-rate"] = score

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
        "--top-p", type=float, default=0.95,
        help="Top-p sampling parameter (nucleus sampling)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples to generate per example"
    )
    parser.add_argument(
        "--eval-data-name", type=str, default="ctrl_RE", choices=["all", "ctrl_RE"],
        help="Dataset name"
    )
    parser.add_argument(
        "--test-set-choice", type=str, default="test_ood-both", choices=["test_id", "test_ood-both"],
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
    lst = []
    for s in samples:
        questions = s[args.question_key]
        for q in questions:
            q["text"] = s["text"]
            q["answer"] = str(q["answer"])
        lst.extend(questions)
    return Dataset.from_list(lst)


def get_messages_from_problem(problem, model_name_or_path_base, dataset_name="ctrl_RE", contexts=None, context_type="no-context"):
    """Extract messages from problem for vLLM API"""
    
    if dataset_name in ["ctrl_RE"]:
        # return [
        #     {"role": "user", "content": problem}
        # ]
        system_prompt = "You are a helpful assistant. Provides a *short* final answer and a confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The *short* final answer is enclosed between <answer> </answer> tags."
        
        if context_type in ["gold-context", "rag-context"]:
            context = "\n\n".join([f"[Document {i}]\n{doc}" for i, doc in enumerate(contexts)]) + "\n\n" + "[Question]\n"
        elif context_type == "no-context":
            context = ""
        elif context_type == "self-ask":
            context = SELF_ASK_PROMPT + "\n\nQuestion: "

        elif context_type == "self-ask_system":
            system_prompt += "\n\n" + SELF_ASK_PROMPT
            context = "Question: "
        elif context_type == "self-ask_musique":
            context = SELF_ASK_PROMPT_MUSIQUE + "\n\nQuestion: "
        elif context_type == "self-ask_system_musique":
            system_prompt += "\n\n" + SELF_ASK_PROMPT_MUSIQUE
            context = "Question: "
        elif context_type == "self-ask_w-instruction":
            system_prompt = "You are a helpful assistant. First decompose the question into a series of follow up questions. Then answer the follow up questions one by one. Provides a *short* final answer and a confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The *short* final answer is enclosed between <answer> </answer> tags."
            context = SELF_ASK_PROMPT + "\n\nQuestion: "
        elif context_type == "self-ask_w-instruction_musique":
            system_prompt = "You are a helpful assistant. First decompose the question into a series of follow up questions. Then answer the follow up questions one by one. Provides a *short* final answer and a confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The *short* final answer is enclosed between <answer> </answer> tags."
            context = SELF_ASK_PROMPT_MUSIQUE + "\n\nQuestion: "
        elif context_type == "self-ask_w-instruction_system_musique":
            system_prompt = "You are a helpful assistant. First decompose the question into a series of follow up questions. Then answer the follow up questions one by one. Provides a *short* final answer and a confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The *short* final answer is enclosed between <answer> </answer> tags.\n\nSee demonstration below.\n\n" + SELF_ASK_PROMPT_MUSIQUE
            context = "Question: "
        elif context_type == "self-ask_w-instruction_system":
            system_prompt = "You are a helpful assistant. First decompose the question into a series of follow up questions. Then answer the follow up questions one by one. Provides a *short* final answer and a confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The *short* final answer is enclosed between <answer> </answer> tags.\n\nSee demonstration below.\n\n" + SELF_ASK_PROMPT
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
        eval_data_names = ["ctrl_RE"]
    else:
        eval_data_names = [args.eval_data_name]
    model = None
    
    for eval_data_name in eval_data_names:
        model_name_or_path_base = os.path.basename(args.model_name_or_path)
        save_dir = f"/u/zliu/datastor1/LLaMA-Factory/eval_saves/{model_name_or_path_base}"
        os.makedirs(save_dir, exist_ok=True)
        output_file = f"{save_dir}/{eval_data_name}_{args.test_set_choice}_{args.context_type}.xlsx"
    
        if not os.path.exists(output_file) or args.overwrite:
            
            sampling_params = SamplingParams(
                max_tokens=args.max_tokens, 
                top_p=args.top_p,
                temperature=args.temperature,
                n=args.num_samples,
                skip_special_tokens=False,
            )
            # import pdb; pdb.set_trace()
            dataset = load_controlled_RE_data(args, f"/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/syn_data_neurips/4K_train_data_100percent_comp/{args.test_set_choice}.jsonl")
            if eval_data_name == "ctrl_RE":
                problem_key = "question"
                answer_key = "answer"
            else:
                raise ValueError(f"Invalid dataset name: {eval_data_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            if args.context_type == "gold-context":
                all_messages = [get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name, contexts=[d["text"]], context_type=args.context_type) for d in dataset]
            elif args.context_type == "rag-context":
                # context = d["text"]
                retriever = DualRetrieverLite(document_texts=list(set([d['text'] for d in dataset])), dense_retriever_name="none", top_k=1, chunk_size=128, chunk_overlap=0)
                all_messages = [get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name, contexts=[r["text"] for r in retriever.query(d[problem_key])], context_type=args.context_type) for d in dataset]
            else:
                # import pdb; pdb.set_trace()
                all_messages = [get_messages_from_problem(d[problem_key], model_name_or_path_base=model_name_or_path_base, dataset_name=eval_data_name, contexts=None, context_type=args.context_type) for d in dataset]
            
            
            # import pdb; pdb.set_trace()
            
            try:
                texts = tokenizer.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                # import pdb; pdb.set_trace()
                if model is None:
                    model = LLM(args.model_name_or_path)
                outputs = model.generate(texts, sampling_params=sampling_params)
                results = []
                
                for idx, output in enumerate(outputs):
                    
                    for sample_idx, out in enumerate(output.outputs):
                        
                        model_answer = out.text
                        # import pdb; pdb.set_trace()
                        model_answer = model_answer.split("</think>")[-1].strip()
                        results.append({
                            "text": dataset[idx]["text"] if eval_data_name in ["ctrl_RE"] else None,
                            "prompt": texts[idx],
                            "question": dataset[idx][problem_key],
                            "eval_data_name": eval_data_name,
                            "ground_truth_answer": dataset[idx][answer_key],
                            "sample_id": sample_idx,
                            "model_answer": model_answer,
                            "model_response": out.text,
                            "question_type": dataset[idx]["type"],
                        })
            except Exception as e:
                print(f"\nError processing: {e}")
            
            
            pd.DataFrame(results).to_excel(output_file, index=False)
            print(f"\nFinal results saved to {output_file}")
        
        # evaluate with llm_judge
        
        for llm_judge_type in ["hard",]:
            df = pd.read_excel(output_file)
            if f"llm_accuracy-{llm_judge_type}" not in df.columns or args.overwrite:
                print(f"Evaluating with [{llm_judge_type}] judge")
                llm_judge_name = args.llm_judge_name
                if llm_judge_type == "hard":
                    llm_judge = LlmAsJudgeHard(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                elif llm_judge_type == "abstention":
                    llm_judge = LlmAsJudgeAbstention(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                else:
                    llm_judge = LlmAsJudge(
                        model_name=llm_judge_name, backend_params={"max_requests_per_minute": 30_000, "max_tokens_per_minute": 150_000_000}
                    )
                df["predicted_answer"] = df["model_response"].astype(str)
                df["answer"] = df["ground_truth_answer"].astype(str)
                scored_dataset = Dataset.from_pandas(df[:])
                scored_dataset = llm_judge(
                    scored_dataset,
                )
                # import pdb; pdb.set_trace()
                ds = scored_dataset
                if hasattr(scored_dataset, "dataset"):
                    ds = scored_dataset.dataset
                    
                scored_df = ds.to_pandas().drop(columns=['predicted_answer', 'answer',])
                scored_df["llm_judge"] = llm_judge_name
                scored_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    main()