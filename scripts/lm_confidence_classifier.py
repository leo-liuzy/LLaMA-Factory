"""
LMConfidenceClassifier: Binary classifier head on top of a left-to-right causal LM.

The model encodes a question with the LM, takes the final hidden state of the
last non-padding token, applies a linear projection to produce a binary logit,
and optimizes with BCE-with-logits loss. The sigmoid(logit) is interpreted as
the model's confidence in being able to answer the question.

Input data formats
- JSONL: each line is {"question": str, "label": 0|1} (label optional for predict)
- CSV: header with columns: question,label (label optional for predict)

Examples
1) Train + evaluate
   python -u /u/zliu/datastor1/LLaMA-Factory/scripts/lm_confidence_classifier.py \
     --model_name_or_path Qwen/Qwen2.5-1.5B \
     --output_dir /u/zliu/datastor1/LLaMA-Factory/saves/conf_classifier_qwen2p5_1p5b \
     --train_file /path/to/train.jsonl \
     --eval_file /path/to/dev.jsonl \
     --batch_size 8 --lr 2e-5 --num_train_epochs 3 --fp16

2) Predict confidences
   python -u /u/zliu/datastor1/LLaMA-Factory/scripts/lm_confidence_classifier.py \
     --model_name_or_path Qwen/Qwen2.5-1.5B \
     --output_dir /u/zliu/datastor1/LLaMA-Factory/saves/conf_classifier_qwen2p5_1p5b \
     --predict_file /path/to/questions.jsonl \
     --batch_size 16

Outputs
- Training checkpoints under output_dir (LM weights + classifier.pt)
- predictions.csv for predict-only runs with columns: question,confidence
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset  # type: ignore[import-not-found]


@dataclass
class TrainConfig:
    model_name_or_path: str
    output_dir: str
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    predict_file: Optional[str] = None
    max_seq_length: int = 1024
    lr: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.0
    dropout: float = 0.0
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    trust_remote_code: bool = False
    # TriviaQA integration
    use_triviaqa: bool = False
    triviaqa_split: str = "train"  # train|validation|test
    triviaqa_version: str = "rc.nocontext"  # "rc.nocontext", "rc", or "unfiltered.nocontext"
    labeling_batch_size: int = 8
    gen_max_new_tokens: int = 16
    gen_temperature: float = 0.0
    gen_top_p: float = 1.0
    gen_do_sample: bool = False
    save_labeled_path: Optional[str] = None


class QuestionsDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        labels = [int(ex["label"]) for ex in batch] if "label" in batch[0] else None
        tokenized = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if labels is not None:
            tokenized["labels"] = torch.tensor(labels, dtype=torch.float32)
        return tokenized


class LMConfidenceClassifier(nn.Module):
    """Wrap a left-to-right causal LM and classify based on the final token state.

    Forward returns a dict with keys: logits (B, 1) and optional loss (scalar).
    """

    def __init__(self, language_model: AutoModelForCausalLM, hidden_dropout: float = 0.0):
        super().__init__()
        self.language_model = language_model
        hidden_size = self._infer_hidden_size(self.language_model)
        self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout and hidden_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, 1)

    @staticmethod
    def _infer_hidden_size(model: AutoModelForCausalLM) -> int:
        # Attempt common locations for hidden size
        if hasattr(model.config, "hidden_size") and model.config.hidden_size is not None:
            return int(model.config.hidden_size)
        if hasattr(model.config, "n_embd") and model.config.n_embd is not None:
            return int(model.config.n_embd)
        if hasattr(model.config, "d_model") and model.config.d_model is not None:
            return int(model.config.d_model)
        # Fallback to probing an input
        raise ValueError("Unable to infer hidden size from model config.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden: torch.Tensor = outputs.hidden_states[-1]
        if attention_mask is None:
            # Assume last token is the last position
            last_indices = torch.full((last_hidden.size(0),), last_hidden.size(1) - 1, device=last_hidden.device, dtype=torch.long)
        else:
            # Last non-padding token index per sequence
            last_indices = attention_mask.long().sum(dim=1) - 1
            last_indices = torch.clamp(last_indices, min=0)

        # Gather final token hidden state
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        final_hidden = last_hidden[batch_indices, last_indices]  # (B, H)
        logits = self.classifier(self.dropout(final_hidden))  # (B, 1)
        result: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            labels = labels.view(-1, 1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss
        return result

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        # Save underlying LM
        self.language_model.save_pretrained(output_dir)
        # Save classifier head
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))

    def load_classifier(self, output_dir: str) -> None:
        path = os.path.join(output_dir, "classifier.pt")
        state = torch.load(path, map_location="cpu")
        self.classifier.load_state_dict(state, strict=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "question" not in obj:
                raise ValueError("JSONL entries must contain 'question'")
            examples.append({"question": obj["question"], **({"label": obj["label"]} if "label" in obj else {})})
    return examples


def read_csv(path: str) -> List[Dict[str, Any]]:
    # Very small CSV reader without external deps, expects header with question,label
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        col_to_idx = {name: i for i, name in enumerate(header)}
        if "question" not in col_to_idx:
            raise ValueError("CSV must contain 'question' column")
        for line in f:
            parts = line.rstrip("\n").split(",")
            question = parts[col_to_idx["question"]]
            entry: Dict[str, Any] = {"question": question}
            if "label" in col_to_idx:
                entry["label"] = int(parts[col_to_idx["label"]])
            examples.append(entry)
    return examples


def load_examples(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    if path.endswith(".csv"):
        return read_csv(path)
    raise ValueError("Unsupported file extension. Use .jsonl or .csv")


def evaluate(model: LMConfidenceClassifier, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"].squeeze(-1)
            if "labels" in batch:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"])  # type: ignore[arg-type]
                total_loss += loss.item() * logits.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == batch.get("labels", preds).long()).sum().item()
            total += logits.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc


def _normalize_answer(text: str) -> str:
    s = text.strip().lower()
    punct = "\'`\".,;:!?()[]{}<>-_/\\|@#$%^&*~+="
    table = str.maketrans({c: " " for c in punct})
    s = s.translate(table)
    s = " ".join(s.split())
    return s


def _extract_gold_answers(example: Dict[str, Any]) -> List[str]:
    ans = example.get("answer", {})
    candidates: List[str] = []
    if isinstance(ans, dict):
        for key in ("normalized_aliases", "aliases"):
            vals = ans.get(key)
            if isinstance(vals, list):
                candidates.extend(vals)
        if ans.get("value") and isinstance(ans["value"], str):
            candidates.append(ans["value"])
    if not candidates and isinstance(example.get("answers"), list):
        candidates.extend([str(a) for a in example["answers"]])
    norm = list({
        _normalize_answer(str(c)) for c in candidates if isinstance(c, str) and len(str(c).strip()) > 0
    })
    return norm


def _label_with_generation(
    language_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    questions: List[str],
    gold_answers: List[List[str]],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    batch_size: int,
) -> List[int]:
    labels: List[int] = []
    language_model.eval()
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        enc = tokenizer(
            batch_q,
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 1024,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            gen = language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_texts: List[str] = []
        for j in range(gen.size(0)):
            inp_len = input_ids[j].size(0)
            out_ids = gen[j][inp_len:]
            gen_texts.append(tokenizer.decode(out_ids, skip_special_tokens=True))
        for j, pred in enumerate(gen_texts):
            norm_pred = _normalize_answer(pred)
            gold_set = set(gold_answers[i + j])
            label = 1 if (norm_pred in gold_set or any(norm_pred == g or norm_pred in g or g in norm_pred for g in gold_set)) else 0
            labels.append(label)
    return labels


def load_triviaqa_and_label(
    language_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    version: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    batch_size: int,
) -> List[Dict[str, Any]]:
    dataset = load_dataset("trivia_qa", version, split=split)
    questions: List[str] = []
    golds: List[List[str]] = []
    for ex in dataset:
        q = ex.get("question")
        if not isinstance(q, str) or len(q.strip()) == 0:
            continue
        questions.append(q)
        golds.append(_extract_gold_answers(ex))
    labels = _label_with_generation(
        language_model=language_model,
        tokenizer=tokenizer,
        questions=questions,
        gold_answers=golds,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        batch_size=batch_size,
    )
    
    examples = [{"question": q, "label": int(y)} for q, y in zip(questions, labels)]
    return examples

def train_and_eval(cfg: TrainConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        # For causal LMs without pad token, fall back to eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None),
        trust_remote_code=cfg.trust_remote_code,
    )
    model = LMConfidenceClassifier(language_model=language_model, hidden_dropout=cfg.dropout).to(device)

    # Datasets
    train_loader: Optional[DataLoader] = None
    eval_loader: Optional[DataLoader] = None
    if cfg.use_triviaqa:
        print(
            f"Loading TriviaQA ({cfg.triviaqa_version}) split={cfg.triviaqa_split} and pseudo-labeling via LM generation..."
        )
        train_examples = load_triviaqa_and_label(
            language_model=language_model,
            tokenizer=tokenizer,
            split=cfg.triviaqa_split,
            version=cfg.triviaqa_version,
            device=device,
            max_new_tokens=cfg.gen_max_new_tokens,
            temperature=cfg.gen_temperature,
            top_p=cfg.gen_top_p,
            do_sample=cfg.gen_do_sample,
            batch_size=cfg.labeling_batch_size,
        )
        if cfg.save_labeled_path:
            os.makedirs(os.path.dirname(cfg.save_labeled_path), exist_ok=True)
            with open(cfg.save_labeled_path, "w", encoding="utf-8") as f:
                for ex in train_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Saved labeled dataset to {cfg.save_labeled_path}")
        train_dataset = QuestionsDataset(train_examples)
        collate = DataCollator(tokenizer, cfg.max_seq_length)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    else:
        if cfg.train_file is None and cfg.predict_file is None:
            raise ValueError("Provide --train_file for training or --predict_file for prediction.")
        if cfg.train_file is not None:
            train_examples = load_examples(cfg.train_file)
            train_dataset = QuestionsDataset(train_examples)
            collate = DataCollator(tokenizer, cfg.max_seq_length)
            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

            if cfg.eval_file is not None:
                eval_examples = load_examples(cfg.eval_file)
                eval_dataset = QuestionsDataset(eval_examples)
                eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    global_step = 0
    if train_loader is not None:
        model.train()
        for epoch in range(cfg.num_train_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=cfg.fp16 or cfg.bf16):
                    outputs = model(**batch)
                    loss = outputs["loss"] / max(cfg.grad_accum_steps, 1)
                scaler.scale(loss).backward()

                if (step + 1) % cfg.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        print(f"epoch={epoch+1} step={global_step} loss={loss.item() * cfg.grad_accum_steps:.4f}")

                    if eval_loader is not None and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                        eval_loss, eval_acc = evaluate(model, eval_loader, device)
                        print(f"[eval] step={global_step} loss={eval_loss:.4f} acc={eval_acc:.4f}")

                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model.save_pretrained(ckpt_dir)

                epoch_loss += loss.item() * cfg.grad_accum_steps
            print(f"epoch {epoch+1} average loss: {epoch_loss / max(len(train_loader), 1):.4f}")

        # Final save
        model.save_pretrained(cfg.output_dir)

        if eval_loader is not None:
            eval_loss, eval_acc = evaluate(model, eval_loader, device)
            print(f"[final eval] loss={eval_loss:.4f} acc={eval_acc:.4f}")

    # Prediction-only path
    if cfg.predict_file is not None:
        predict_examples = load_examples(cfg.predict_file)
        predict_dataset = QuestionsDataset(predict_examples)
        collate = DataCollator(tokenizer, cfg.max_seq_length)
        predict_loader = DataLoader(predict_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

        # Try load classifier head if present in output_dir
        try:
            model.load_classifier(cfg.output_dir)
        except Exception:
            pass

        model.eval()
        all_scores: List[float] = []
        with torch.no_grad():
            for batch in predict_loader:
                batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
                logits = model(**batch)["logits"].squeeze(-1)
                probs = torch.sigmoid(logits).tolist()
                all_scores.extend([float(p) for p in probs])

        # Write out CSV with question and confidence
        out_path = os.path.join(cfg.output_dir, "predictions.csv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("question,confidence\n")
            for ex, score in zip(predict_examples, all_scores):
                # naive escaping of commas by replacing with space
                q = str(ex["question"]).replace(",", " ")
                f.write(f"{q},{score:.6f}\n")
        print(f"Predictions written to: {out_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Binary classifier on top of a causal LM using final token state")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--predict_file", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--trust_remote_code", action="store_true")
    # TriviaQA flags
    parser.add_argument("--use_triviaqa", action="store_true")
    parser.add_argument("--triviaqa_split", type=str, default="train")
    parser.add_argument("--triviaqa_version", type=str, default="rc.nocontext")
    parser.add_argument("--labeling_batch_size", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=16)
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument("--gen_top_p", type=float, default=1.0)
    parser.add_argument("--gen_do_sample", action="store_true")
    parser.add_argument("--save_labeled_path", type=str, default=None)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_and_eval(config)


