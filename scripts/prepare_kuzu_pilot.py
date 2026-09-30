import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


TEXT2CYPHER_REVISION = "d9f15541ab2f99a0f54797a0109b50663554f512"
TEXT2CYPHER_TRAIN_URL = (
    f"https://huggingface.co/datasets/neo4j/text2cypher-2024v1/resolve/"
    f"{TEXT2CYPHER_REVISION}/data/train-00000-of-00001.parquet"
)
TEXT2CYPHER_TEST_URL = (
    f"https://huggingface.co/datasets/neo4j/text2cypher-2024v1/resolve/"
    f"{TEXT2CYPHER_REVISION}/data/test-00000-of-00001.parquet"
)
DOCRED_DIR = Path("data/open_pilots/docred_sft_5k_distant_v1")
TEXT2CYPHER_URL = "https://huggingface.co/datasets/neo4j/text2cypher-2024v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="data/cache/kuzu-pilot")
    parser.add_argument("--docred-train", type=int, default=3500)
    parser.add_argument("--cypher-train", type=int, default=1500)
    parser.add_argument("--docred-val", type=int, default=350)
    parser.add_argument("--cypher-val", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1820)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["question", "schema", "cypher", "instance_id"])
    return table.to_pylist()


def cypher_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "cypher_generation",
        "instruction": (
            "Practice text-to-Cypher using the provided graph schema and question. "
            "Return only the query and preserve the source dataset's Neo4j Cypher dialect."
        ),
        "input": f"Schema:\n{row['schema']}\n\nQuestion:\n{row['question']}",
        "target": str(row["cypher"]).strip(),
        "source_dataset": "neo4j/text2cypher-2024v1",
        "source_license": "apache-2.0 dataset card; component-source terms require review",
        "source_id": str(row["instance_id"]),
    }


def mix_rows(
    docred: list[dict[str, Any]],
    cypher: list[dict[str, Any]],
    docred_count: int,
    cypher_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(docred) < docred_count or len(cypher) < cypher_count:
        raise ValueError("available examples are fewer than requested subset counts")
    docred_rows = rng.sample(docred, docred_count)
    cypher_rows = [cypher_example(row) for row in rng.sample(cypher, cypher_count)]
    rows = docred_rows + cypher_rows
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    docred_train_path = DOCRED_DIR / "train.jsonl"
    docred_val_path = DOCRED_DIR / "val.jsonl"
    if not docred_train_path.exists() or not docred_val_path.exists():
        raise FileNotFoundError(f"DocRED pilot data missing under {DOCRED_DIR}")

    cypher_train_path = fetch(
        TEXT2CYPHER_TRAIN_URL, cache_dir / f"text2cypher-{TEXT2CYPHER_REVISION}-train.parquet"
    )
    cypher_test_path = fetch(
        TEXT2CYPHER_TEST_URL, cache_dir / f"text2cypher-{TEXT2CYPHER_REVISION}-test.parquet"
    )
    cypher_train = parquet_rows(cypher_train_path)
    cypher_test = parquet_rows(cypher_test_path)

    train_rows = mix_rows(
        read_jsonl(docred_train_path), cypher_train,
        args.docred_train, args.cypher_train, args.seed,
    )
    val_rows = mix_rows(
        read_jsonl(docred_val_path), cypher_test,
        args.docred_val, args.cypher_val, args.seed + 1,
    )
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    manifest = {
        "seed": args.seed,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_task_counts": {
            "graph_extraction": args.docred_train,
            "cypher_generation": args.cypher_train,
        },
        "val_task_counts": {
            "graph_extraction": args.docred_val,
            "cypher_generation": args.cypher_val,
        },
        "docred_source": "thunlp/docred train_distant split, MIT",
        "docred_train_sha256": sha256_file(docred_train_path),
        "docred_val_sha256": sha256_file(docred_val_path),
        "text2cypher_source": TEXT2CYPHER_URL,
        "text2cypher_revision": TEXT2CYPHER_REVISION,
        "text2cypher_license": "Apache-2.0 on dataset card; source component terms not independently reviewed",
        "text2cypher_train_url": TEXT2CYPHER_TRAIN_URL,
        "text2cypher_test_url": TEXT2CYPHER_TEST_URL,
        "text2cypher_train_sha256": sha256_file(cypher_train_path),
        "text2cypher_test_sha256": sha256_file(cypher_test_path),
        "train_sha256": sha256_file(output_dir / "train.jsonl"),
        "val_sha256": sha256_file(output_dir / "val.jsonl"),
        "limitation": "Cypher rows retain Neo4j syntax and are not yet validated against Kuzu; mix covers two capabilities only.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
