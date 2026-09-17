import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=64)
    parser.add_argument("--max-val-examples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def prompt_text(row: dict[str, Any]) -> str:
    return (
        "Extract entities and relations from the chunk. "
        "Return JSON with entities, relations, and claims.\n\n"
        f"Chunk:\n{row['input']}\n\nJSON:\n"
    )


def completion_text(row: dict[str, Any]) -> str:
    return json.dumps(row["target"], ensure_ascii=False, sort_keys=True)


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, torch.Tensor]:
    prompt = prompt_text(row)
    completion = completion_text(row) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    min_prompt_tokens = min(64, max(1, max_length // 4))
    completion_budget = max_length - min_prompt_tokens
    if len(completion_ids) > completion_budget:
        completion_ids = completion_ids[: completion_budget - 1] + [tokenizer.eos_token_id]
    prompt_budget = max_length - len(completion_ids)
    if prompt_budget <= 0:
        raise ValueError("max_length leaves no room for supervised completion tokens")
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + completion_ids
    labels = input_ids[:]
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        raise ValueError("encoded example has no supervised labels")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
        out["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
    return {key: torch.stack(value) for key, value in out.items()}


def evaluate(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available. Refusing to run a SmolLM2 training smoke on CPU. "
            "Pass --allow-cpu only for a tiny proxy-model loop check."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(args.train_jsonl), args.max_train_examples)
    val_rows = read_jsonl(Path(args.val_jsonl), args.max_val_examples)
    train_data = [encode_row(tokenizer, row, args.max_length) for row in train_rows]
    val_data = [encode_row(tokenizer, row, args.max_length) for row in val_rows]
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = {
        "status": "running",
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "losses": [],
        "note": "Diagnostic loss smoke only. Not a gate metric.",
    }
    start = time.time()
    initial_val = evaluate(model, val_loader, device)
    metrics["initial_val_loss"] = initial_val

    for epoch in range(args.epochs):
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate(model, val_loader, device)
        metrics["losses"].append({
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "val_loss": val_loss,
        })

    final_val = metrics["losses"][-1]["val_loss"] if metrics["losses"] else initial_val
    metrics["final_val_loss"] = final_val
    metrics["val_loss_delta"] = final_val - initial_val
    metrics["perplexity_initial"] = math.exp(min(initial_val, 20))
    metrics["perplexity_final"] = math.exp(min(final_val, 20))
    metrics["elapsed_seconds"] = round(time.time() - start, 3)
    metrics["status"] = "complete"
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
