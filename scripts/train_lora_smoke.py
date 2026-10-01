import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--model-revision", default="a10cc1512eabd3dde888204e902eca88bddb4951")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--source-license", default=None)
    parser.add_argument("--source-manifest", default=None)
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
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--allow-world-size-migration", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--max-run-seconds", type=int, default=0)
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
    if row.get("instruction"):
        return f"{row['instruction'].rstrip()}\n\nInput:\n{row['input']}\n\nOutput:\n"
    return (
        "Extract entities and relations from the chunk. "
        "Return JSON with entities, relations, and claims.\n\n"
        f"Chunk:\n{row['input']}\n\nJSON:\n"
    )


def completion_text(row: dict[str, Any]) -> str:
    target = row["target"]
    if isinstance(target, str):
        return target
    return json.dumps(target, ensure_ascii=False, sort_keys=True)


def encode_row(
    tokenizer: Any, row: dict[str, Any], max_length: int, use_chat_template: bool
) -> dict[str, torch.Tensor | bool]:
    prompt = prompt_text(row)
    completion = completion_text(row) + tokenizer.eos_token
    if use_chat_template:
        if not tokenizer.chat_template:
            raise ValueError("--chat-template requested but tokenizer has no chat template")
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    min_prompt_tokens = min(96, max(1, max_length // 3))
    completion_budget = max_length - min_prompt_tokens
    completion_truncated = len(completion_ids) > completion_budget
    if completion_truncated:
        completion_ids = completion_ids[: completion_budget - 1] + [tokenizer.eos_token_id]
    prompt_budget = max_length - len(completion_ids)
    if prompt_budget <= 0:
        raise ValueError("max_length leaves no room for supervised completion tokens")

    prompt_truncated = len(prompt_ids) > prompt_budget
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
        "completion_truncated": completion_truncated,
        "prompt_truncated": prompt_truncated,
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


def batches_for_epoch(data: list[dict[str, torch.Tensor]], batch_size: int, seed: int,
                      epoch: int, rank: int = 0, world_size: int = 1,
                      global_rows_consumed: int = 0) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(data), generator=generator).tolist()
    global_batch = batch_size * world_size
    padded_size = math.ceil(len(order) / global_batch) * global_batch
    if padded_size > len(order):
        order.extend(order[:padded_size - len(order)])
    if global_rows_consumed % global_batch:
        raise ValueError("Resume cursor is not aligned to the distributed microbatch")
    batches = []
    for start in range(global_rows_consumed, len(order), global_batch):
        local_start = start + rank * batch_size
        batches.append(order[local_start:local_start + batch_size])
    return batches


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_signature(args: argparse.Namespace, train_path: Path, val_path: Path) -> dict[str, Any]:
    signature = {
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "qlora": args.qlora,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
        "attn_implementation": args.attn_implementation,
        "chat_template": args.chat_template,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
    }
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        signature["world_size"] = world_size
    return signature


def resolve_checkpoint(value: str, output_dir: Path) -> Path:
    def is_complete(candidate: Path) -> bool:
        adapter_dir = candidate / "adapter"
        return (
            (candidate / "state.pt").is_file()
            and (adapter_dir / "adapter_config.json").is_file()
            and any((adapter_dir / name).is_file() for name in [
                "adapter_model.safetensors", "adapter_model.bin"
            ])
        )

    if value == "latest":
        candidates = sorted(
            (output_dir / "checkpoints").glob("step-*"),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            if is_complete(candidate):
                return candidate
        raise FileNotFoundError("no complete checkpoint is available to resume")
    checkpoint = Path(value)
    if not is_complete(checkpoint):
        raise FileNotFoundError(f"checkpoint adapter is incomplete: {checkpoint}")
    return checkpoint


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scaler: Any,
    output_dir: Path,
    signature: dict[str, Any],
    epoch: int,
    next_batch_idx: int,
    optimizer_steps: int,
    args: argparse.Namespace,
    train_input_tokens_seen: int,
    epoch_train_losses: list[float],
    elapsed_seconds: float,
    rank: int = 0,
    world_size: int = 1,
    global_rows_consumed: int = 0,
) -> None:
    rng_by_rank: list[dict[str, Any] | None] = [None] * world_size
    local_rng = rng_state()
    if world_size > 1:
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank[0] = local_rng
    if rank != 0:
        dist.barrier()
        return

    checkpoints_dir = output_dir / "checkpoints"
    checkpoint_dir = checkpoints_dir / f"step-{optimizer_steps:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.save_pretrained(checkpoint_dir / "adapter")
    state = {
        "signature": signature,
        "epoch": epoch,
        "next_batch_idx": next_batch_idx,
        "global_rows_consumed": global_rows_consumed,
        "optimizer_steps": optimizer_steps,
        "train_input_tokens_seen": train_input_tokens_seen,
        "epoch_train_losses": epoch_train_losses,
        "elapsed_seconds": elapsed_seconds,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_by_rank": rng_by_rank,
        "rng": rng_by_rank[0],
    }
    state_path = checkpoint_dir / "state.pt"
    temporary_state = checkpoint_dir / "state.pt.tmp"
    torch.save(state, temporary_state)
    os.replace(temporary_state, state_path)
    write_json(checkpoint_dir / "manifest.json", {
        "epoch": epoch,
        "next_batch_idx": next_batch_idx,
        "optimizer_steps": optimizer_steps,
        "signature": signature,
    })
    checkpoints = sorted(
        (path for path in checkpoints_dir.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for old_checkpoint in checkpoints[:-args.keep_last_checkpoints]:
        shutil.rmtree(old_checkpoint)
    if world_size > 1:
        dist.barrier()


def evaluate(model: Any, batches: list[dict[str, torch.Tensor]], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
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


def length_summary(rows: list[dict[str, torch.Tensor]]) -> dict[str, int | float]:
    lengths = sorted(len(row["input_ids"]) for row in rows)
    if not lengths:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int((len(lengths) - 1) * 0.95)],
        "max": lengths[-1],
    }


def main() -> None:
    args = parse_args()
    session_start = time.time()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the local SmolLM2 LoRA smoke run.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(f"torchrun requested {world_size} GPUs; only {torch.cuda.device_count()} are visible")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    environment = env_info()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        args.resume_from = str(resolve_checkpoint(args.resume_from, output_dir))
    if is_main:
        append_log(output_dir, f"run started world_size={world_size} local_batch={args.batch_size} grad_accum={args.grad_accum_steps}")

    if args.resume_from:
        for required_file in ["config.yaml", "env.json", "metrics.json"]:
            if not (output_dir / required_file).exists():
                raise FileNotFoundError(f"cannot resume without existing {required_file}")
    elif is_main:
        config_record = vars(args).copy()
        config_record["note"] = "Diagnostic local QLoRA/LoRA smoke only. Not a gate metric."
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config_record, sort_keys=True),
            encoding="utf-8",
        )
        write_json(output_dir / "env.json", environment)
    if distributed:
        dist.barrier()

    if is_main:
        append_log(output_dir, f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        append_log(output_dir, "reading data")
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    train_rows = read_jsonl(train_path, args.max_train_examples)
    val_rows = read_jsonl(val_path, args.max_val_examples)
    train_data = [encode_row(tokenizer, row, args.max_length, args.chat_template) for row in train_rows]
    val_data = [encode_row(tokenizer, row, args.max_length, args.chat_template) for row in val_rows]
    val_task_data: dict[str, list[dict[str, Any]]] = {}
    for row, encoded in zip(val_rows, val_data, strict=True):
        task = row.get("task", "graph_extraction")
        val_task_data.setdefault(task, []).append(encoded)

    val_batches = [
        collate(val_data[index:index + args.batch_size], tokenizer.pad_token_id)
        for index in range(0, len(val_data), args.batch_size)
    ]
    val_batches_by_task = {
        task: [
            collate(rows[index:index + args.batch_size], tokenizer.pad_token_id)
            for index in range(0, len(rows), args.batch_size)
        ]
        for task, rows in val_task_data.items()
    }

    if is_main:
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
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        **model_kwargs,
    )
    active_attention = getattr(base_model.config, "_attn_implementation", None)
    if active_attention != args.attn_implementation:
        raise RuntimeError(
            f"requested attention {args.attn_implementation}, active attention is {active_attention}"
        )
    base_model.config.use_cache = False
    if args.qlora:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    else:
        base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.resume_from:
        model = PeftModel.from_pretrained(
            base_model, str(Path(args.resume_from) / "adapter"), is_trainable=True
        )
    else:
        model = get_peft_model(base_model, lora_config)
    if not args.qlora:
        model.to(device)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )
    model.train()
    param_counts = count_parameters(model)
    if is_main:
        append_log(output_dir, f"trainable params {param_counts['trainable_parameter_count']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    signature = checkpoint_signature(args, train_path, val_path)

    metrics: dict[str, Any] = {
        "status": "running",
        "note": "Diagnostic local QLoRA/LoRA smoke only. Not a gate metric.",
        "model": args.model,
        "model_revision": args.model_revision,
        "source_dataset": args.source_dataset,
        "source_license": args.source_license,
        "source_manifest": (
            json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
            if args.source_manifest else None
        ),
        "source_train_sha256": sha256_file(train_path),
        "source_val_sha256": sha256_file(val_path),
        "gold_set_version": None,
        "gold_set_sha256": None,
        "gold_set_examples": 0,
        "training_mode": "qlora" if args.qlora else "lora",
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type if args.qlora else None,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant if args.qlora else None,
        "seed": args.seed,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "max_length": args.max_length,
        "chat_template": args.chat_template,
        "attn_implementation_requested": args.attn_implementation,
        "attn_implementation_active": active_attention,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "world_size": world_size,
        "effective_global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        **param_counts,
        "losses": [],
        "progress": [],
        "resume_events": [],
        "task_counts": {
            task: sum(row.get("task", "graph_extraction") == task for row in train_rows)
            for task in sorted({row.get("task", "graph_extraction") for row in train_rows})
        },
        "train_sequence_length_tokens": length_summary(train_data),
        "val_sequence_length_tokens": length_summary(val_data),
        "train_prompt_truncated_examples": sum(row["prompt_truncated"] for row in train_data),
        "train_completion_truncated_examples": sum(row["completion_truncated"] for row in train_data),
        "val_prompt_truncated_examples": sum(row["prompt_truncated"] for row in val_data),
        "val_completion_truncated_examples": sum(row["completion_truncated"] for row in val_data),
    }
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    if is_main:
        append_log(output_dir, "evaluating initial validation loss")
    prior_train_tokens_seen = 0
    local_train_tokens_seen = 0
    resume_global_rows = 0
    previous_elapsed = 0.0
    time_limit_reached = False
    try:
        metrics_path = output_dir / "metrics.json"
        if args.resume_from:
            checkpoint_state = torch.load(
                Path(args.resume_from) / "state.pt", map_location="cpu", weights_only=False
            )
            saved_signature = checkpoint_state["signature"]
            migration = saved_signature != signature
            if migration:
                old_world = int(saved_signature.get("world_size", 1))
                old_effective_batch = (
                    int(saved_signature["batch_size"])
                    * int(saved_signature["grad_accum_steps"])
                    * old_world
                )
                new_effective_batch = args.batch_size * args.grad_accum_steps * world_size
                ignored = {"grad_accum_steps", "world_size"}
                compatible = (
                    args.allow_world_size_migration
                    and old_world == 1 and world_size > 1
                    and old_effective_batch == new_effective_batch
                    and {k: v for k, v in saved_signature.items() if k not in ignored}
                    == {k: v for k, v in signature.items() if k not in ignored}
                )
                if not compatible:
                    raise ValueError("resume checkpoint does not match the current training recipe or data")
            optimizer.load_state_dict(checkpoint_state["optimizer"])
            scaler.load_state_dict(checkpoint_state["scaler"])
            rng_states = checkpoint_state.get("rng_by_rank", [checkpoint_state["rng"]])
            restore_rng(rng_states[rank] if rank < len(rng_states) else rng_state())
            step = checkpoint_state["optimizer_steps"]
            start_epoch = checkpoint_state["epoch"]
            resume_batch_idx = checkpoint_state["next_batch_idx"]
            prior_train_tokens_seen = checkpoint_state["train_input_tokens_seen"]
            resume_global_rows = int(checkpoint_state.get(
                "global_rows_consumed", resume_batch_idx * args.batch_size
            ))
            resumed_epoch_losses = checkpoint_state["epoch_train_losses"]
            previous_elapsed = checkpoint_state.get("elapsed_seconds", 0.0)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["status"] = "running"
            metrics.pop("stop_reason", None)
            metrics.pop("paused_at_optimizer_step", None)
            if is_main:
                metrics.setdefault("resume_events", []).append({
                    "checkpoint": Path(args.resume_from).name,
                    "optimizer_steps": step,
                    "epoch": start_epoch + 1,
                    "next_batch_idx": resume_batch_idx,
                    "global_rows_consumed": resume_global_rows,
                    "world_size": world_size,
                    "effective_global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
                    "world_size_migration": migration,
                })
                metrics["distributed_world_size"] = world_size
                metrics["effective_global_batch_size"] = args.batch_size * args.grad_accum_steps * world_size
                metrics["batch_size"] = args.batch_size
                metrics["grad_accum_steps"] = args.grad_accum_steps
                write_json(metrics_path, metrics)
                append_log(output_dir, f"resumed optimizer_step {step} epoch {start_epoch + 1} batch {resume_batch_idx} world_size={world_size}")
            if distributed:
                dist.barrier()
            initial_val = metrics["initial_val_loss"]
        else:
            initial_val = evaluate(model, val_batches, device)
            metrics["initial_val_loss"] = initial_val
            metrics["initial_val_loss_by_task"] = {
                task: evaluate(model, batches, device)
                for task, batches in val_batches_by_task.items()
            }
            step = 0
            start_epoch = 0
            resume_batch_idx = 0
            resume_global_rows = 0
            resumed_epoch_losses = []
            if is_main:
                metrics["distributed_world_size"] = world_size
                metrics["effective_global_batch_size"] = args.batch_size * args.grad_accum_steps * world_size
                metrics["batch_size"] = args.batch_size
                metrics["grad_accum_steps"] = args.grad_accum_steps
                write_json(metrics_path, metrics)
            if distributed:
                dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            train_losses = resumed_epoch_losses if epoch == start_epoch else []
            optimizer.zero_grad(set_to_none=True)
            if is_main:
                append_log(output_dir, f"epoch {epoch + 1} started")
            epoch_start_rows = resume_global_rows if epoch == start_epoch else 0
            epoch_batches = batches_for_epoch(
                train_data, args.batch_size, args.seed, epoch,
                rank, world_size, epoch_start_rows,
            )
            for batch_idx, row_indices in enumerate(epoch_batches):
                batch = collate([train_data[index] for index in row_indices], tokenizer.pad_token_id)
                local_train_tokens_seen += int(batch["attention_mask"].sum().item())
                batch = {key: value.to(device) for key, value in batch.items()}
                accumulation_start = batch_idx - batch_idx % args.grad_accum_steps
                accumulation_size = min(
                    args.grad_accum_steps, len(epoch_batches) - accumulation_start
                )
                should_step = (
                    (batch_idx + 1) % args.grad_accum_steps == 0
                    or batch_idx + 1 == len(epoch_batches)
                )
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model(**batch).loss / accumulation_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, batch {batch_idx + 1}")
                    scaler.scale(loss).backward()
                train_loss = float((loss * accumulation_size).detach().cpu())
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"non-finite train loss at epoch {epoch + 1}, batch {batch_idx + 1}")
                if distributed:
                    loss_total = torch.tensor(train_loss, device=device)
                    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
                    train_loss = float(loss_total.item() / world_size)
                train_losses.append(train_loss)

                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    time_limit_reached = (
                        args.max_run_seconds > 0
                        and time.time() - session_start >= args.max_run_seconds
                    )
                    if distributed:
                        stop_flag = torch.tensor(int(time_limit_reached), device=device)
                        dist.broadcast(stop_flag, src=0)
                        time_limit_reached = bool(stop_flag.item())
                    checkpoint_due = (
                        args.checkpoint_every_steps > 0
                        and step % args.checkpoint_every_steps == 0
                    )
                    if checkpoint_due or time_limit_reached:
                        global_token_count = torch.tensor(local_train_tokens_seen, device=device, dtype=torch.long)
                        if distributed:
                            dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
                        global_train_tokens_seen = prior_train_tokens_seen + int(global_token_count.item())
                        global_rows_consumed = epoch_start_rows + (batch_idx + 1) * args.batch_size * world_size
                        save_checkpoint(
                            model, optimizer, scaler, output_dir, signature,
                            epoch, global_rows_consumed // args.batch_size, step, args,
                            global_train_tokens_seen, train_losses,
                            previous_elapsed + time.time() - start,
                            rank, world_size, global_rows_consumed,
                        )
                    should_log = (
                        args.log_every_steps > 0 and step % args.log_every_steps == 0
                    ) or time_limit_reached
                    if should_log:
                        global_token_count = torch.tensor(local_train_tokens_seen, device=device, dtype=torch.long)
                        if distributed:
                            dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
                        global_train_tokens_seen = prior_train_tokens_seen + int(global_token_count.item())
                        elapsed = previous_elapsed + time.time() - start
                        window_start = max(0, len(train_losses) - args.log_every_steps * args.grad_accum_steps)
                        recent_loss = sum(train_losses[window_start:]) / max(
                            len(train_losses[window_start:]), 1
                        )
                        metrics["optimizer_steps"] = step
                        metrics["train_input_tokens_seen"] = global_train_tokens_seen
                        metrics["elapsed_seconds_so_far"] = round(elapsed, 3)
                        metrics["train_input_tokens_per_second_wall_so_far"] = round(
                            global_train_tokens_seen / max(elapsed, 1e-9),
                            3,
                        )
                        if torch.cuda.is_available() and is_main:
                            metrics["cuda_max_memory_allocated_bytes_so_far"] = torch.cuda.max_memory_allocated()
                            metrics["cuda_max_memory_reserved_bytes_so_far"] = torch.cuda.max_memory_reserved()
                        progress = {
                            "epoch": epoch + 1,
                            "optimizer_step": step,
                            "recent_train_loss": recent_loss,
                            "elapsed_seconds": round(elapsed, 3),
                            "train_input_tokens_seen": global_train_tokens_seen,
                            "train_input_tokens_per_second_wall": round(
                                global_train_tokens_seen / max(elapsed, 1e-9), 3
                            ),
                        }
                        if is_main:
                            metrics["progress"].append(progress)
                            append_log(output_dir, json.dumps(progress, sort_keys=True))
                        if time_limit_reached:
                            metrics["status"] = "paused"
                            metrics["stop_reason"] = "max_run_seconds"
                            metrics["paused_at_optimizer_step"] = step
                            if is_main:
                                append_log(output_dir, f"time limit reached; checkpoint saved at optimizer_step {step}")
                        if is_main:
                            write_json(output_dir / "metrics.json", metrics)
                    if time_limit_reached:
                        break

            if time_limit_reached:
                break

            val_loss = evaluate(model, val_batches, device)
            val_loss_by_task = {
                task: evaluate(model, batches, device)
                for task, batches in val_batches_by_task.items()
            }
            metrics["losses"].append({
                "epoch": epoch + 1,
                "optimizer_steps": step,
                "train_loss": sum(train_losses) / max(len(train_losses), 1),
                "val_loss": val_loss,
                "val_loss_by_task": val_loss_by_task,
            })
            epoch_token_count = torch.tensor(local_train_tokens_seen, device=device, dtype=torch.long)
            if distributed:
                dist.all_reduce(epoch_token_count, op=dist.ReduceOp.SUM)
            global_train_tokens_seen = prior_train_tokens_seen + int(epoch_token_count.item())
            if is_main:
                append_log(output_dir, f"epoch {epoch + 1} val_loss {val_loss}")
                write_json(output_dir / "metrics.json", metrics)
            save_checkpoint(
                model, optimizer, scaler, output_dir, signature,
                epoch + 1, 0, step, args,
                global_train_tokens_seen, [],
                previous_elapsed + time.time() - start,
                rank, world_size, 0,
            )
            resume_batch_idx = 0
            resume_global_rows = 0

        if time_limit_reached:
            if is_main:
                append_log(output_dir, f"run paused at optimizer_step {step}; resume from latest checkpoint")
        else:
            final_val = metrics["losses"][-1]["val_loss"] if metrics["losses"] else initial_val
            metrics["status"] = "complete"
            metrics["final_val_loss"] = final_val
            metrics["val_loss_delta"] = final_val - initial_val
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure_type"] = type(exc).__name__
        metrics["failure_message"] = str(exc)
        if is_main:
            append_log(output_dir, f"run failed {type(exc).__name__}: {exc}")
        raise
    finally:
        elapsed = time.time() - start
        final_token_count = torch.tensor(local_train_tokens_seen, device=device, dtype=torch.long)
        if distributed:
            dist.all_reduce(final_token_count, op=dist.ReduceOp.SUM)
        global_train_tokens_seen = prior_train_tokens_seen + int(final_token_count.item())
        metrics["elapsed_seconds_current_session"] = round(elapsed, 3)
        metrics["elapsed_seconds"] = round(previous_elapsed + elapsed, 3)
        metrics["train_input_tokens_seen"] = global_train_tokens_seen
        metrics["train_input_tokens_per_second_wall"] = round(global_train_tokens_seen / max(elapsed, 1e-9), 3)
        if torch.cuda.is_available():
            metrics["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            metrics["cuda_mem_free_bytes_at_end"] = free
            metrics["cuda_mem_total_bytes_at_end"] = total
        if is_main:
            write_json(output_dir / "metrics.json", metrics)

    if args.save_adapter and is_main:
        adapter_dir = output_dir / "adapter"
        unwrapped_model = model.module if hasattr(model, "module") else model
        unwrapped_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    if metrics["status"] == "complete" and is_main:
        append_log(output_dir, "run complete")
    if is_main:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
