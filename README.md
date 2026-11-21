# The Graph Changes, the Answers Don't: 4-Bit Quantization in Knowledge-Graph RAG

Code, frozen configurations and result summaries for the paper, accepted at the **6th Workshop on
Knowledge Graphs and Big Data (KGBigdata 2026)**, IEEE BigData 2026, Phoenix, AZ (paper ID S05203).

Mirza Shahzaib Baig, Tobias Deußer, Lorenz Sparrenberg, Rafet Sifa — University of Bonn, Lamarr
Institute, Fraunhofer IAIS.

## What the study does

A KG-RAG pipeline invokes its generator twice: once to **extract** a knowledge graph from the corpus
at index time, once to **answer** from that graph at query time. Quantizing the generator can
therefore degrade quality through a channel that standard benchmarks never probe — a worse graph that
handicaps every later query.

We test this with a controlled paired experiment in LightRAG. Documents, chunking, embedder, retrieval
mode, decoding parameters and judge are held fixed; only the generator's weight precision varies
(F16 vs. 4-bit `Q4_K_M` Qwen2.5-7B-Instruct). 50 indexes across five SQuAD corpus sizes × five seeds,
33,497 paired questions.

The graphs diverge (exact node overlap 0.55–0.62, edge 0.23–0.27) while every downstream measure is
indistinguishable (|d_z| ≤ 0.031). Semantic overlap reaches 0.85 for nodes against 0.62 exact, so most
divergence is surface form; on the residual, each arm answers ~0.11/5 better on questions touching
entities only it extracted (p < 1e-7 both directions), so the advantages cancel.

## What is and is not in this repository

**Included:** every script in the pipeline, the SLURM job scripts used on our cluster, the aggregate
result summaries the paper's numbers come from (~2.7 MB), the build/eval timing log, and the paper
source together with the generators that produce all of its tables and figures.

**Not included,** by size, with the command that regenerates each:

| Artifact | Size | Regenerate with |
|---|---|---|
| Built LightRAG indexes (`SQUAD/{f16,q4}/…`, `TRIVIAQA/…`) | ~17 GB | the build scripts below |
| Frozen corpora (`data/*.jsonl`) | ~109 MB | `freeze_squad_train_multi.py`, `freeze_triviaqa.py` |
| Per-question evaluation dumps | ~67 MB | the eval scripts below |
| Node-exclusive question subsets | ~36 MB | `node_exclusive_questions.py` |
| Model weights (`*.gguf`) | ~20 GB | download from Hugging Face (below) |

**LightRAG is not vendored.** We used it unmodified at **v1.4.9.9** (commit `dccf1ef4`); only its
configuration and the evaluation scripts around it are ours. Install it from PyPI or check out that
commit from `HKUDS/LightRAG`.

## Setup

```bash
pip install -r requirements.txt          # or requirements-a100.lock.txt for our exact pins
```

Our cluster is heterogeneous, which is why two environments appear in the SLURM scripts: an
AVX-512 build for Intel (A40) nodes and an AVX2-safe build for AMD (A100) nodes. A single-machine
reproduction needs only one.

### Models

All LLMs are served locally by [`llama.cpp`](https://github.com/ggml-org/llama.cpp) over its
OpenAI-compatible HTTP server. Two servers run during evaluation — the generator under study and the
judge:

| Port | Model | Role |
|---|---|---|
| 8001 | Qwen2.5-7B-Instruct **F16** | generator, F16 arm |
| 8000 | Qwen2.5-7B-Instruct **Q4_K_M** | generator, Q4 arm |
| 8005 / 8004 | gpt-oss-20b (F16) | judge for the F16 / Q4 eval |

Embeddings use `Qwen/Qwen3-Embedding-0.6B` through SentenceTransformer (1024-d, L2-normalised) at
**full fp32 precision, never quantized** — it is a control, shared identically by both arms. Builds
select their generator with the `LLM_BASE_URL` environment variable.

## Reproducing the pipeline

Run every command from the repository root; the scripts resolve paths relative to it.

**1 — Freeze the corpora.** Sampling is seeded and written to disk so every build and evaluation reads
byte-identical data.

```bash
python freeze_squad_train_multi.py     # data/squad_train_{size}_v{0..4}.jsonl
python freeze_triviaqa.py              # data/triviaqa_docs_{25,50,75,100}.jsonl
```

**2 — Build the indexes.** One index per (arm, size, seed); 50 for SQuAD.

```bash
python build_lightrag_kg_index_qwenf16_multi.py --size 500 --version 0   # -> SQUAD/f16/500/v0/
python build_lightrag_kg_index_qwenQ4_multi.py  --size 500 --version 0   # -> SQUAD/q4/500/v0/
python build_lightrag_kg_index_qwenf16_triviaqa_multi.py --size 50 [--resume]
```

SQuAD chunking is 600/100, TriviaQA 300/50. The TriviaQA builders deliberately run at low concurrency
with retries and `--resume`, because its evidence documents are full Wikipedia articles (~29 chunks)
and high concurrency caused long-document extraction failures.

**3 — Evaluate.**

```bash
python eval_squad_multi_f16.py --size 500 --version 0    # judged correctness
python eval_squad_multi_q4.py  --size 500 --version 0
python lexical_eval.py   --size 500 [--bertscore]        # Cover-EM / token recall
python retrieval_eval.py --size 500 --index-model f16 --version 0   # Hit@k / MRR
```

**4 — Compare and analyse.**

```bash
python score_compare_multi.py        # paired Wilcoxon + Cohen's d_z + rank-biserial
python eval_KG_multi.py --sizes 200 500 1000 2000 3000 --semantic
python compute_semantic_edge.py      # semantic edge overlap
python node_exclusive_questions.py   # tag questions by model-exclusive entity
python analyze_node_exclusive.py     # the symmetric double dissociation
bash slurm/collect_timings.sh        # -> build_eval_timings.csv
```

**5 — Rebuild the paper.** Every table and figure is generated from the result files; none is typed by
hand. Edit the generator, never `paper/tables/*.tex`.

```bash
python paper/make_paper_tables.py    # -> paper/tables/table_0{1..6}.tex
python paper/make_paper_figures.py   # -> paper/pics/*.pdf  (pins SOURCE_DATE_EPOCH; idempotent)
cd paper && tectonic -X compile main.tex
```

### The TriviaQA replication

Section VI of the paper reports a single-seed generalization check on TriviaQA, whose evidence
documents are complete Wikipedia articles rather than single paragraphs. It uses parallel scripts
throughout, at document counts 25/50/75/100:

```bash
python build_lightrag_kg_index_qwenf16_triviaqa_multi.py --size 50 [--resume]
python build_lightrag_kg_index_qwenQ4_triviaqa_multi.py  --size 50 [--resume]
python eval_triviaqa_f16.py --size 50
python eval_triviaqa_q4.py  --size 50
python lexical_eval_triviaqa.py --size 50
python retrieval_eval_triviaqa.py --size 50 --index-model f16
python score_compare_triviaqa.py     # -> TRIVIAQA/results/paired_triviaqa.csv
python eval_KG_triviaqa.py --semantic # -> KG_results/kg_triviaqa_comparison.json
```

### On a SLURM cluster

`slurm/` holds the job scripts used for the reported runs. Per-seed job arrays keep each job inside the
scheduler's limits:

```bash
sbatch --array=0-4 --export=ALL,SIZE=500 slurm/slurm_build_f16.sh
sbatch --array=0-4 --export=ALL,SIZE=500 slurm/slurm_build_Q4.sh
sbatch --array=0-4 --exclusive --job-name=eval_f16_500 --export=ALL,SIZE=500 slurm/slurm_eval_f16.sh
sbatch --array=0-4 --exclusive --job-name=eval_q4_500  --export=ALL,SIZE=500 slurm/slurm_eval_q4.sh
sbatch --array=0-4 --job-name=retr_f16_500 --export=ALL,SIZE=500,IDX_MODEL=f16 slurm/slurm_retrieval.sh
```

| Script | Purpose |
|---|---|
| `slurm_build_{f16,Q4}.sh` | SQuAD index builds, one seed per array task |
| `slurm_build_f16_short.sh` | same, for a short-queue partition with uncapped concurrency |
| `slurm_build_{f16,q4}_a100.sh` | A100 variants (AVX2-safe environment) |
| `slurm_build_triviaqa_{f16,q4}.sh` | TriviaQA builds; low concurrency plus `RESUME=1` for long articles |
| `slurm_eval_{f16,q4}.sh` | evaluation, two GPUs per job (generator + judge) |
| `slurm_eval_triviaqa_{f16,q4}.sh` | the TriviaQA equivalents |
| `slurm_retrieval{,_triviaqa}.sh` | Hit@k / MRR scoring |
| `collect_timings.sh` | parses job logs into `build_eval_timings.csv` |

These encode our cluster's partitions, QOS limits and module layout. Treat them as a reference for the
job shapes rather than as portable scripts.

## Results included

| Path | Contents |
|---|---|
| `SQUAD/results/` | per-seed and cross-seed correctness, lexical and retrieval summaries; paired statistics; node-exclusive summary |
| `KG_results/` | `kg_multi_comparison.json`, structural and overlap metrics per size |
| `TRIVIAQA/results/` | the single-seed generalization check |
| `build_eval_timings.csv` | parsed SLURM wall-clock for builds and evaluations |

## Scale and runtime

The reported runs cost several hundred GPU-hours on NVIDIA A40s. A single F16 build at 3000 documents
exceeds a 24-hour job limit, which is what sets the upper end of the corpus-size range. Q4 builds are
2.34× faster at size 200 and 2.21× at 500, narrowing to 1.14× at 2000 as embedding, merging and I/O
come to dominate. Weight memory is 4.68 GB against 15.24 GB.

If you want to reproduce a slice rather than the whole study, size 200 with one seed is the cheapest
cell that exercises every stage.

## Known scope

The result characterises KG-RAG whose retrieval reaches the raw chunks alongside the graph, on
predominantly single-hop questions. A pipeline that answers by traversing the graph itself (text-to-Cypher,
or retrieval dominated by graph structure) has no chunk-level fallback for a mislabelled entity or a
missing edge and may be considerably more sensitive; we have not tested that class of system. The study
also does not quantify how much the graph contributes relative to chunk retrieval in the first place —
a `naive`-mode ablation over these same indexes is the direct test. See the paper's Limitations.

## Citation

```bibtex
@inproceedings{baig2026graphchanges,
  title     = {The Graph Changes, the Answers Don't: 4-Bit Quantization in Knowledge-Graph {RAG}},
  author    = {Baig, Mirza Shahzaib and Deu{\ss}er, Tobias and Sparrenberg, Lorenz and Sifa, Rafet},
  booktitle = {Proceedings of the 6th Workshop on Knowledge Graphs and Big Data (KGBigdata),
               IEEE International Conference on Big Data (BigData)},
  year      = {2026}
}
```

## License

**Not yet chosen — add a `LICENSE` file before publishing.** MIT or Apache-2.0 are the usual choices
for an artifact repository; confirm with your institution and co-authors, since Fraunhofer and Lamarr
affiliations can carry their own requirements. Note that SQuAD (CC BY-SA 4.0), TriviaQA (Apache-2.0),
Qwen2.5 and gpt-oss each carry their own licenses, which this repository does not relicense.
