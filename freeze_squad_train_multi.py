"""
Freeze SQuAD (train split) into fixed multi-seed slices for the F16-vs-Q4 study.
For each size in {200, 500, 1000, 2000, 3000, 4000, 8000} and seed i in 0..4, take
shuffle(seed=i).select(range(size)) and write it to data/squad_train_{size}_v{i}.jsonl,
so that every build and evaluation reads exactly the same data.

Usage:  python freeze_squad_train_multi.py
"""
from datasets import load_dataset

NUM_SETS = 5
SIZES = [200, 500, 1000, 2000, 3000, 4000, 8000]

ds = load_dataset("squad", split="train")

for size in SIZES:
    for i in range(NUM_SETS):
        subset = ds.shuffle(seed=i).select(range(size))
        subset.to_json(f"./data/squad_train_{size}_v{i}.jsonl")
        print(f"size={size} v{i}: saved {len(subset)} examples (seed={i})")
