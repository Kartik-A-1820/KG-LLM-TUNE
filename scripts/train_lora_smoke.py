import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--source-license", default=None)
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
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=10)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    packages = {}
    for package in ["transformers", "peft", "accelerate", "bitsandbytes", "trl"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "commit_sha": git_value(["rev-parse", "HEAD"]),
        "dirty_tree": bool(git_value(["status", "--short"])),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": packages,
        "cuda_available": cuda,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_mem_free_bytes_at_start": free,
        "cuda_mem_total_bytes": total,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(output_dir: Path, message: str) -> None:
    with (output_dir / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def count_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "parameter_count": total,
        "trainable_parameter_count": trainable,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the local SmolLM2 LoRA smoke run.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    environment = env_info()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "run started")

    config_record = vars(args).copy()
    config_record["note"] = "Diagnostic local QLoRA/LoRA smoke only. Not a gate metric."
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config_record, sort_keys=True),
        encoding="utf-8",
    )
    write_json(output_dir / "env.json", environment)

    append_log(output_dir, f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    append_log(output_dir, "reading data")
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    train_rows = read_jsonl(train_path, args.max_train_examples)
    val_rows = read_jsonl(val_path, args.max_val_examples)
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

    append_log(output_dir, f"loading model qlora={args.qlora}")
    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
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
    if not args.qlora:
        model.to(device)
    model.train()
    param_counts = count_parameters(model)
    append_log(output_dir, f"trainable params {param_counts['trainable_parameter_count']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    metrics: dict[str, Any] = {
        "status": "running",
        "note": "Diagnostic local QLoRA/LoRA smoke only. Not a gate metric.",
        "model": args.model,
        "source_dataset": args.source_dataset,
        "source_license": args.source_license,
        "source_train_sha256": sha256_file(train_path),
        "source_val_sha256": sha256_file(val_path),
        "training_mode": "qlora" if args.qlora else "lora",
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type if args.qlora else None,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant if args.qlora else None,
        "seed": args.seed,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        **param_counts,
        "losses": [],
    }
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    append_log(output_dir, "evaluating initial validation loss")
    train_input_tokens_seen = 0
    try:
        initial_val = evaluate(model, val_loader, device)
        metrics["initial_val_loss"] = initial_val

        step = 0
        for epoch in range(args.epochs):
            train_losses = []
            optimizer.zero_grad(set_to_none=True)
            append_log(output_dir, f"epoch {epoch + 1} started")
            for batch_idx, batch in enumerate(train_loader):
                train_input_tokens_seen += int(batch["attention_mask"].sum().item())
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(**batch).loss / args.grad_accum_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {batch_idx + 1}")
                scaler.scale(loss).backward()
                train_loss = float((loss * args.grad_accum_steps).detach().cpu())
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"non-finite train loss at epoch {epoch + 1}, batch {batch_idx + 1}")
                train_losses.append(train_loss)

                should_step = (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx + 1 == len(train_loader)
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if args.log_every_steps > 0 and step % args.log_every_steps == 0:
                        elapsed = time.time() - start
                        metrics["optimizer_steps"] = step
                        metrics["train_input_tokens_seen"] = train_input_tokens_seen
                        metrics["elapsed_seconds_so_far"] = round(elapsed, 3)
                        metrics["train_input_tokens_per_second_wall_so_far"] = round(
                            train_input_tokens_seen / max(elapsed, 1e-9),
                            3,
                        )
                        if torch.cuda.is_available():
                            metrics["cuda_max_memory_allocated_bytes_so_far"] = torch.cuda.max_memory_allocated()
                            metrics["cuda_max_memory_reserved_bytes_so_far"] = torch.cuda.max_memory_reserved()
                        append_log(output_dir, f"optimizer_step {step}")
                        write_json(output_dir / "metrics.json", metrics)

            val_loss = evaluate(model, val_loader, device)
            metrics["losses"].append({
                "epoch": epoch + 1,
                "optimizer_steps": step,
                "train_loss": sum(train_losses) / max(len(train_losses), 1),
                "val_loss": val_loss,
            })
            append_log(output_dir, f"epoch {epoch + 1} val_loss {val_loss}")
            write_json(output_dir / "metrics.json", metrics)

        final_val = metrics["losses"][-1]["val_loss"] if metrics["losses"] else initial_val
        metrics["status"] = "complete"
        metrics["final_val_loss"] = final_val
        metrics["val_loss_delta"] = final_val - initial_val
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure_type"] = type(exc).__name__
        metrics["failure_message"] = str(exc)
        append_log(output_dir, f"run failed {type(exc).__name__}: {exc}")
        raise
    finally:
        elapsed = time.time() - start
        metrics["elapsed_seconds"] = round(elapsed, 3)
        metrics["train_input_tokens_seen"] = train_input_tokens_seen
        metrics["train_input_tokens_per_second_wall"] = round(train_input_tokens_seen / max(elapsed, 1e-9), 3)
        if torch.cuda.is_available():
            metrics["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            metrics["cuda_mem_free_bytes_at_end"] = free
            metrics["cuda_mem_total_bytes_at_end"] = total
        write_json(output_dir / "metrics.json", metrics)

    if args.save_adapter:
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    append_log(output_dir, "run complete")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
