# FOLD

Code for the paper *FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding*.

FOLD lets block-parallel drafters such as DFlash and DSpark, which are trained on short sequences, handle long contexts without any training. Recent context keeps its exact relative positions, while all older context is read as a far memory at one shared position, normalized separately from the recent window. The drafter's weights, the target model and verification are unchanged.

This repository provides the core implementation of FOLD as a vLLM plugin, for reference.

## Repository structure

```
fold/
  attention.py     FOLD attention for vLLM drafters, with the far-memory side cache
  kernels.py       Triton kernels for the far-memory read and the merge
  reference.py     plain PyTorch version of FOLD attention
  vllm_plugin.py   installs FOLD in the drafter when FOLD_CONFIG is set
  rope.py, rope_cache.py
scripts/
  run.py           AR or speculative decoding, records per-request TPOT and acceptance length
  summarize.py     speedup over AR with bootstrap confidence intervals
patches/
  vllm-0.28.0-dspark.patch   needed for the DeepSpec drafters
```

## Installation

The code targets vLLM 0.28.0 on a single GPU (we used an H100).

```bash
pip install -e .
export CUDA_HOME=/path/to/cuda   # must match the CUDA version of PyTorch
```

`CUDA_HOME` is needed because vLLM compiles its sampling kernels at startup.

The DeepSpec drafters of Gemma-4-12B and Qwen3-8B also need a small patch to vLLM:

```bash
patch -p1 -d "$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parents[1])')" < patches/vllm-0.28.0-dspark.patch
```

## Usage

`scripts/run.py` reads tokenized requests, one JSON object per line, with the prompt already formatted by the target's chat template:

```json
{"id": "0", "task": "gov", "input_ids": [128000, ...], "max_new_tokens": 512}
```

`task` is optional. `summarize.py` reports each task and averages tasks with equal weight.

Example with Llama-3.1-8B:

```bash
COMMON="--target NousResearch/Meta-Llama-3.1-8B-Instruct --max-model-len 131072 --inputs requests.jsonl"
SD="--drafter z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --method dflash --num-speculative-tokens 9 --draft-max-position 131072"

python scripts/run.py $COMMON --output out/ar.jsonl                                 # AR baseline
python scripts/run.py $COMMON $SD --output out/dflash.jsonl                         # original drafter
python scripts/run.py $COMMON $SD --fold-window 2048 --output out/fold.jsonl        # FOLD

python scripts/summarize.py --ar out/ar.jsonl --method DFlash=out/dflash.jsonl --method FOLD=out/fold.jsonl
```

Settings used in the paper (α = 2 for all pairs):

| Target | Drafter | `--method` | `--num-speculative-tokens` | `--max-model-len` | `--fold-window` | Extra options |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | z-lab DFlash | dflash | 9 | 131072 | 2048 | `--draft-max-position 131072` |
| Qwen3.5-9B | z-lab DFlash | dflash | 15 | 262144 | 4096 | `--fold-sliding-layers fold` |
| Gemma-4-12B-it | DeepSpec DFlash / DSpark | dspark | 7 | 262144 | 4096 | `--fold-local-splits 64 --deterministic-inductor` |
| Qwen3-8B | DeepSpec DFlash / DSpark | dspark | 7 | 32768 | 2048 | |

The checkpoints are listed in the paper's appendix. Generation stops at the target's end-of-turn tokens by default (`--stop-token-ids` overrides them). `--fold-mode window` runs FOLD without its far memory, and `--fold-mode rerope` runs the ReRoPE baseline on the same attention path.

## Notes

- Supported setting: batch size 1, BF16, FlashAttention 3 or 4, and the DFlash and DSpark drafters of vLLM 0.28.0.
- The far memory is stored in a side cache whose size grows linearly with the context length (up to 6.0 GiB for the Qwen3.5-9B drafter at 262K tokens, or 1.0 GiB with `--fold-sliding-layers window`).
- `patches/vllm-0.28.0-dspark.patch` adapts vLLM (Apache 2.0) so that it loads the DeepSpec DFlash checkpoints and runs the DeepSpec drafters as in their reference implementation. Model weights and datasets are not included.
