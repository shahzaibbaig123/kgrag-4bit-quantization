"""
Find the questions where the F16 and Q4 arms diverge most in judged correctness,
for the qualitative case-study section of the thesis (Ch.7).

Offline / read-only over existing results:
  - per-question (question, gold, answer, score) from SQUAD/results/eval_squad_{size}_v{v}_qwen{f16,Q4}.txt
  - node-exclusive group tags from node_exclusive_qa/subset_{size}_v{v}.jsonl (if present)

Output: SQUAD/results/divergent_cases.json  (top cases per direction, sorted by |diff|)
Prints direction counts as a symmetry check.

Run:  python find_divergent_cases.py
"""
import os, re, json
from collections import Counter

SQ = "SQUAD/results"
NE = "node_exclusive_qa"
SIZES = [200, 500, 1000, 2000, 3000]
TOP_PER_DIR = 15

_BLOCK = re.compile(
    r"#\d+\nQ:\s*(?P<q>.*?)\nGold:\s*(?P<gold>.*?)\nAnswer:\s*(?P<ans>.*?)\nScore:\s*(?P<score>[\d.]+)",
    re.DOTALL)


def parse_eval(path):
    out = {}
    if not os.path.exists(path):
        return out
    for m in _BLOCK.finditer(open(path, encoding="utf-8").read()):
        ans = m.group("ans").split("### References")[0].strip()
        out[m.group("q").strip()] = (m.group("gold").strip(), ans, float(m.group("score")))
    return out


def ne_groups(size, v):
    """question -> node-exclusive group (f16_only / q4_only / both / control)."""
    p = f"{NE}/subset_{size}_v{v}.jsonl"
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["question"].strip()] = r.get("group", "?")
    return out


def main():
    cases = []
    dir_counts = Counter()      # sign of diff for |diff| >= 3
    total_pairs = 0
    for size in SIZES:
        for v in range(5):
            f = parse_eval(f"{SQ}/eval_squad_{size}_v{v}_qwenf16.txt")
            q = parse_eval(f"{SQ}/eval_squad_{size}_v{v}_qwenQ4.txt")
            ne = ne_groups(size, v)
            common = set(f) & set(q)
            total_pairs += len(common)
            for k in common:
                d = f[k][2] - q[k][2]
                if abs(d) >= 3:
                    dir_counts["f16" if d > 0 else "q4"] += 1
                    cases.append({
                        "size": size, "seed": v, "diff": d,
                        "question": k, "gold": f[k][0],
                        "f16_score": f[k][2], "q4_score": q[k][2],
                        "f16_answer": f[k][1], "q4_answer": q[k][1],
                        "ne_group": ne.get(k, "none"),
                    })
    cases.sort(key=lambda c: -abs(c["diff"]))
    f16_wins = [c for c in cases if c["diff"] > 0][:TOP_PER_DIR]
    q4_wins = [c for c in cases if c["diff"] < 0][:TOP_PER_DIR]
    out = {
        "total_paired_questions": total_pairs,
        "n_diverging_ge3": sum(dir_counts.values()),
        "direction_counts_ge3": dict(dir_counts),
        "f16_wins": f16_wins,
        "q4_wins": q4_wins,
    }
    dst = f"{SQ}/divergent_cases.json"
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print(f"paired questions: {total_pairs}")
    print(f"|diff| >= 3: {sum(dir_counts.values())}  (F16 higher: {dir_counts['f16']}, Q4 higher: {dir_counts['q4']})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
