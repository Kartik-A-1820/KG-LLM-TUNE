import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


RELATION_LABELS = [
    "worked_with",
    "wrote_algorithm_for",
    "wrote_notes_in",
    "based_in",
    "acquired",
    "acquired_in",
    "co_founder_of",
    "located_in",
    "built",
    "built_for",
    "born_in",
    "worked_in",
    "discovered",
    "named_after",
    "released",
    "released_in",
    "built_on",
    "invested_in",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


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
    for package in ["transformers", "accelerate", "bitsandbytes", "torch"]:
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


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def prompt_for(passage: str) -> str:
    labels = ", ".join(RELATION_LABELS)
    return (
        "You extract a knowledge graph from a passage.\n"
        "Return only valid JSON with exactly these top-level keys: entities, relations, claims.\n"
        "Entity objects must have id, name, type, description.\n"
        "Relation objects must have head, tail, type, description. Use entity ids for head and tail.\n"
        f"Allowed relation type values: {labels}.\n"
        "Only include relations directly supported by the passage. Do not add explanations.\n\n"
        f"Passage:\n{passage}\n\nJSON:\n"
    )


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
    return None, "json_parse_failed"


def norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def entity_map(payload: dict[str, Any]) -> dict[str, str]:
    out = {}
    for entity in payload.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("id", "")).strip()
        name = str(entity.get("name", "")).strip()
        if entity_id and name:
            out[entity_id] = name
    return out


def relation_triples(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    id_to_name = entity_map(payload)
    triples = set()
    for relation in payload.get("relations", []):
        if not isinstance(relation, dict):
            continue
        head_raw = str(relation.get("head", "")).strip()
        tail_raw = str(relation.get("tail", "")).strip()
        relation_type = str(relation.get("type", "")).strip()
        head = id_to_name.get(head_raw, head_raw)
        tail = id_to_name.get(tail_raw, tail_raw)
        if head and tail and relation_type:
            triples.add((norm(head), norm(relation_type), norm(tail)))
    return triples


def validate_shape(payload: dict[str, Any]) -> list[str]:
    errors = []
    for key in ["entities", "relations", "claims"]:
        if key not in payload:
            errors.append(f"missing_{key}")
        elif not isinstance(payload[key], list):
            errors.append(f"{key}_not_list")
    ids = set()
    for entity in payload.get("entities", []):
        if isinstance(entity, dict) and entity.get("id"):
            ids.add(str(entity["id"]))
    for relation in payload.get("relations", []):
        if not isinstance(relation, dict):
            errors.append("relation_not_object")
            continue
        if str(relation.get("head", "")) not in ids:
            errors.append("relation_head_missing")
        if str(relation.get("tail", "")) not in ids:
            errors.append("relation_tail_missing")
        if str(relation.get("type", "")) not in RELATION_LABELS:
            errors.append("relation_type_out_of_inventory")
    return errors


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this local relation extraction smoke run.")

    torch.manual_seed(args.seed)
    environment = env_info()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "run started")

    config_record = vars(args).copy()
    config_record["note"] = "Diagnostic open relation extraction smoke only. Not a gate metric."
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config_record, sort_keys=True), encoding="utf-8")
    write_json(output_dir / "env.json", environment)

    examples = read_jsonl(Path(args.input_jsonl), args.max_examples)
    predictions_path = output_dir / "predictions.jsonl"
    start = time.time()
    metrics: dict[str, Any] = {
        "status": "running",
        "note": "Diagnostic open relation extraction smoke only. Not a gate metric.",
        "model": args.model,
        "input_jsonl": args.input_jsonl,
        "examples": len(examples),
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation_requested": args.attn_implementation,
    }

    try:
        append_log(output_dir, f"loading tokenizer {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "attn_implementation": args.attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["dtype"] = torch.float16

        append_log(output_dir, f"loading model {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if not args.load_in_4bit:
            model.to("cuda")
        model.eval()
        metrics["attn_implementation_active"] = getattr(model.config, "_attn_implementation", None)

        parse_success = 0
        schema_valid = 0
        exact_correct = 0
        predicted_total = 0
        gold_total = 0
        schema_error_counts: dict[str, int] = {}

        with predictions_path.open("w", encoding="utf-8", newline="\n") as pred_handle:
            for idx, example in enumerate(examples, start=1):
                append_log(output_dir, f"example {idx} started")
                prompt = prompt_for(example["passage"])
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to("cuda")
                input_tokens = int(inputs["attention_mask"].sum().item())
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                output_ids = generated[0, inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                parsed, parse_error = extract_json(raw_output)
                gold_payload = example["target"]
                gold_triples = relation_triples(gold_payload)
                pred_triples = relation_triples(parsed) if parsed else set()
                correct = pred_triples & gold_triples
                errors = validate_shape(parsed) if parsed else [parse_error or "json_parse_failed"]

                if parsed:
                    parse_success += 1
                if parsed and not errors:
                    schema_valid += 1
                for error in errors:
                    schema_error_counts[error] = schema_error_counts.get(error, 0) + 1

                exact_correct += len(correct)
                predicted_total += len(pred_triples)
                gold_total += len(gold_triples)

                pred_handle.write(json.dumps({
                    "id": example.get("id"),
                    "input_tokens": input_tokens,
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "gold_triples": sorted(gold_triples),
                    "predicted_triples": sorted(pred_triples),
                    "correct_triples": sorted(correct),
                    "schema_errors": errors,
                }, ensure_ascii=False, sort_keys=True))
                pred_handle.write("\n")

        precision = safe_div(exact_correct, predicted_total)
        recall = safe_div(exact_correct, gold_total)
        f1 = safe_div(2 * exact_correct, predicted_total + gold_total)
        metrics.update({
            "status": "complete",
            "parse_success_count": parse_success,
            "schema_valid_count": schema_valid,
            "schema_error_counts": schema_error_counts,
            "gold_relation_triples": gold_total,
            "predicted_relation_triples": predicted_total,
            "correct_relation_triples_strict": exact_correct,
            "strict_relation_precision": precision,
            "strict_relation_recall": recall,
            "strict_relation_f1": f1,
            "predictions_path": str(predictions_path),
        })
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure_type"] = type(exc).__name__
        metrics["failure_message"] = str(exc)
        append_log(output_dir, f"run failed {type(exc).__name__}: {exc}")
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

    append_log(output_dir, "run complete")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
