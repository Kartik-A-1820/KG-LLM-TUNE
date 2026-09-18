import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=32)
    parser.add_argument("--max-val-examples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-adapter", action="store_true")
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

    min_prompt_tokens = min(96, max(1, max_length // 3))
    completion_budget = max_length - min_prompt_tokens
    if len(completion_ids) > completion_budget:
        completion_ids = completion_ids[: completion_budget - 1] + [tokenizer.eos_token_id]
    prompt_budget = max_length - len(completion_ids)
    if prompt_budget <= 0:
        raise ValueError("max_length leaves no room for supervised completion tokens")

    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + completion_ids
    labels = input_ids[:]
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
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


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def env_info() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    free = total = None
    if cuda:
        free, total = torch.cuda.mem_get_info()
    return {
        "commit_sha": git_value(["rev-parse", "HEAD"]),
        "dirty_tree": bool(git_value(["status", "--short"])),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_mem_free_bytes_at_start": free,
        "cuda_mem_total_bytes": total,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the local SmolLM2 LoRA smoke run.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_record = vars(args).copy()
    config_record["note"] = "Diagnostic local LoRA smoke only. Not a gate metric."
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config_record, sort_keys=True),
        encoding="utf-8",
    )
    write_json(output_dir / "env.json", env_info())

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    metrics: dict[str, Any] = {
        "status": "running",
        "note": "Diagnostic local LoRA smoke only. Not a gate metric.",
        "model": args.model,
        "seed": args.seed,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lora_r": args.lora_r,
        "losses": [],
    }
    start = time.time()
    initial_val = evaluate(model, val_loader, device)
    metrics["initial_val_loss"] = initial_val

    step = 0
    for epoch in range(args.epochs):
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            train_losses.append(float((loss * args.grad_accum_steps).detach().cpu()))

            should_step = (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx + 1 == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        val_loss = evaluate(model, val_loader, device)
        metrics["losses"].append({
            "epoch": epoch + 1,
            "optimizer_steps": step,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "val_loss": val_loss,
        })
        write_json(output_dir / "metrics.json", metrics)

    final_val = metrics["losses"][-1]["val_loss"] if metrics["losses"] else initial_val
    metrics["status"] = "complete"
    metrics["final_val_loss"] = final_val
    metrics["val_loss_delta"] = final_val - initial_val
    metrics["elapsed_seconds"] = round(time.time() - start, 3)
    if torch.cuda.is_available():
        metrics["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()

    if args.save_adapter:
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
