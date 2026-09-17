import argparse
import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://huggingface.co/datasets/thunlp/docred/resolve/main/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--source-cache", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if path.exists():
        return path

    url = f"{BASE_URL}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def entity_name(mentions: list[dict[str, Any]]) -> str:
    if not mentions:
        return ""
    return str(mentions[0].get("name", "")).strip()


def entity_type(mentions: list[dict[str, Any]]) -> str:
    if not mentions:
        return ""
    return str(mentions[0].get("type", "")).strip()


def doc_text(example: dict[str, Any]) -> str:
    sentences = []
    for sent in example["sents"]:
        sentences.append(" ".join(sent))
    return "\n".join(sentences)


def convert_example(example: dict[str, Any]) -> dict[str, Any] | None:
    vertex_set = example["vertexSet"]
    labels = example.get("labels", [])
    if isinstance(labels, dict):
        label_iter = []
        for i in range(len(labels.get("head", []))):
            label_iter.append({
                "h": labels["head"][i],
                "t": labels["tail"][i],
                "r": labels["relation_text"][i],
            })
    else:
        label_iter = labels

    entities = []
    for idx, mentions in enumerate(vertex_set):
        name = entity_name(mentions)
        if not name:
            continue
        entities.append({
            "id": f"e{idx}",
            "name": name,
            "type": entity_type(mentions),
            "description": name,
        })

    relations = []
    for label in label_iter:
        head = int(label.get("h", label.get("head", -1)))
        tail = int(label.get("t", label.get("tail", -1)))
        relation_type = str(label.get("r", label.get("relation_text", ""))).strip()
        if head < 0 or tail < 0 or not relation_type:
            continue
        if head >= len(vertex_set) or tail >= len(vertex_set):
            continue
        relations.append({
            "head": f"e{head}",
            "tail": f"e{tail}",
            "type": relation_type,
            "description": relation_type,
        })

    if not entities or not relations:
        return None

    target = {
        "entities": entities,
        "relations": relations,
        "claims": [],
    }
    return {
        "source_dataset": "thunlp/docred",
        "source_license": "mit",
        "source_title": example.get("title", ""),
        "input": doc_text(example),
        "target": target,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def subset(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:limit]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.source_cache) if args.source_cache else output_dir / "source"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_source = fetch("train_annotated.json.gz", cache_dir)
    val_source = fetch("dev.json.gz", cache_dir)

    train_raw = read_gzip_json(train_source)
    val_raw = read_gzip_json(val_source)
    train_rows = [row for row in (convert_example(item) for item in train_raw) if row]
    val_rows = [row for row in (convert_example(item) for item in val_raw) if row]

    train_out = subset(train_rows, args.max_train, args.seed)
    val_out = subset(val_rows, args.max_val, args.seed + 1)

    write_jsonl(output_dir / "train.jsonl", train_out)
    write_jsonl(output_dir / "val.jsonl", val_out)

    manifest = {
        "source": "thunlp/docred",
        "source_url": "https://huggingface.co/datasets/thunlp/docred",
        "license": "mit",
        "seed": args.seed,
        "raw_train_examples": len(train_raw),
        "raw_val_examples": len(val_raw),
        "converted_train_examples": len(train_rows),
        "converted_val_examples": len(val_rows),
        "written_train_examples": len(train_out),
        "written_val_examples": len(val_out),
        "dropped_train_examples": len(train_raw) - len(train_rows),
        "dropped_val_examples": len(val_raw) - len(val_rows),
        "train_source_sha256": sha256_file(train_source),
        "val_source_sha256": sha256_file(val_source),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
