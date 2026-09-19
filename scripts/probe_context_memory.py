import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--supervised-tail-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_versions() -> dict[str, str | None]:
    packages = {}
    for package in ["transformers", "peft", "accelerate", "bitsandbytes", "trl"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return packages


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
        "packages": package_versions(),
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


def synthetic_batch(tokenizer: Any, batch_size: int, max_length: int, supervised_tail_tokens: int) -> dict[str, torch.Tensor]:
    if supervised_tail_tokens <= 0 or supervised_tail_tokens > max_length:
        raise ValueError("supervised_tail_tokens must be in 1..max_length")
    token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("tokenizer needs eos_token_id or pad_token_id")

    input_ids = torch.full((batch_size, max_length), int(token_id), dtype=torch.long)
    attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : max_length - supervised_tail_tokens] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the local context memory probe.")

    torch.manual_seed(args.seed)
    environment = env_info()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "probe started")

    config_record = vars(args).copy()
    config_record["note"] = "Synthetic local context memory probe only. Not a quality or gate metric."
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config_record, sort_keys=True), encoding="utf-8")
    write_json(output_dir / "env.json", environment)

    metrics: dict[str, Any] = {
        "status": "running",
        "note": "Synthetic local context memory probe only. Not a quality or gate metric.",
        "model": args.model,
        "training_mode": "qlora" if args.qlora else "lora",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "supervised_tail_tokens": args.supervised_tail_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "attn_implementation_requested": args.attn_implementation,
        "seed": args.seed,
    }
    start = time.time()

    try:
        append_log(output_dir, f"loading config {args.model}")
        model_config = AutoConfig.from_pretrained(args.model)
        metrics["native_max_position_embeddings"] = getattr(model_config, "max_position_embeddings", None)
        if metrics["native_max_position_embeddings"] is not None and args.max_length > metrics["native_max_position_embeddings"]:
            raise ValueError("max_length exceeds model native max_position_embeddings")

        append_log(output_dir, f"loading tokenizer {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "attn_implementation": args.attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if args.qlora:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["dtype"] = torch.float16

        append_log(output_dir, f"loading model qlora={args.qlora}")
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        model.config.use_cache = False
        metrics["attn_implementation_active"] = getattr(model.config, "_attn_implementation", None)
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
            model.to("cuda")
        model.train()
        metrics.update(count_parameters(model))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda")
        batch = synthetic_batch(tokenizer, args.batch_size, args.max_length, args.supervised_tail_tokens)
        batch = {key: value.to("cuda") for key, value in batch.items()}

        torch.cuda.reset_peak_memory_stats()
        append_log(output_dir, "running forward_backward_optimizer_step")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite probe loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        metrics["status"] = "complete"
        metrics["loss"] = float(loss.detach().cpu())
    except torch.cuda.OutOfMemoryError as exc:
        metrics["status"] = "failed"
        metrics["failure_type"] = "OutOfMemoryError"
        metrics["failure_message"] = str(exc)
        append_log(output_dir, f"probe failed OutOfMemoryError: {exc}")
        raise
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure_type"] = type(exc).__name__
        metrics["failure_message"] = str(exc)
        append_log(output_dir, f"probe failed {type(exc).__name__}: {exc}")
        raise
    finally:
        metrics["elapsed_seconds"] = round(time.time() - start, 3)
        if torch.cuda.is_available():
            metrics["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            metrics["cuda_mem_free_bytes_at_end"] = free
            metrics["cuda_mem_total_bytes_at_end"] = total
        write_json(output_dir / "metrics.json", metrics)

    append_log(output_dir, "probe complete")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
