import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora_smoke import batches_for_epoch


class DistributedBatchTests(unittest.TestCase):
    def test_two_rank_shards_reconstruct_global_batches(self):
        rows = [{} for _ in range(11)]
        shards = [batches_for_epoch(rows, 1, 1820, 0, rank, 2) for rank in range(2)]
        flattened = [value for pair in zip(*shards, strict=True) for shard in pair for value in shard]
        expected = torch.randperm(len(rows), generator=torch.Generator().manual_seed(1820)).tolist()
        self.assertEqual(flattened[:len(rows)], expected)
        self.assertEqual(len(shards[0]), len(shards[1]))

    def test_resume_cursor_starts_both_ranks_after_same_global_rows(self):
        rows = [{} for _ in range(12)]
        full = batches_for_epoch(rows, 1, 1820, 0, 0, 1)
        shards = [batches_for_epoch(rows, 1, 1820, 0, rank, 2, 4) for rank in range(2)]
        resumed = [value for pair in zip(*shards, strict=True) for shard in pair for value in shard]
        self.assertEqual(resumed, [batch[0] for batch in full[4:]])


if __name__ == "__main__":
    unittest.main()
