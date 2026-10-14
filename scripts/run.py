"""Run AR decoding or speculative decoding (optionally with FOLD) over a set of
requests and record per-request TPOT and acceptance length.

Batch size 1, greedy decoding, chunked prefill of 8,192 tokens, CUDA graphs.

Input JSONL, one request per line:
    {"id": "...", "task": "...", "input_ids": [...], "max_new_tokens": 512}
Output JSONL, one record per request with output_ids, ttft, tpot (seconds,
decode time after the first output token divided by the remaining tokens) and
tau (1 + accepted draft tokens / verification rounds).

See README.md for the command lines of each target-drafter pair.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SPEC_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True)
    p.add_argument("--drafter", help="omit for target-only AR decoding")
    p.add_argument("--method", choices=["dflash", "dspark"], default="dflash",
                   help="vLLM speculative method (DeepSpec DFlash checkpoints run as dspark)")
    p.add_argument("--num-speculative-tokens", type=int)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--stop-token-ids", type=int, nargs="+",
                   help="default: eos_token_id of the target's generation config")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draft-max-position", type=int, default=0,
                   help="enlarge the RoPE cache of a DFlash drafter to this length")
    p.add_argument("--deterministic-inductor", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--warmup-tokens", type=int, default=64)
    p.add_argument("--warmup-runs", type=int, default=2)
    g = p.add_argument_group("FOLD (enabled by --fold-window)")
    g.add_argument("--fold-window", type=int, default=0, help="W")
    g.add_argument("--fold-alpha", type=float, default=2.0)
    g.add_argument("--fold-mode", choices=["fold", "window", "rerope"], default="fold")
    g.add_argument("--fold-sliding-layers", choices=["window", "fold"], default="window")
    g.add_argument("--fold-logn-ref", type=int, default=4096, help="ReRoPE log-n reference length")
    g.add_argument("--fold-local-splits", type=int, default=16)
    args = p.parse_args()
    if args.drafter and not args.num_speculative_tokens:
        p.error("--num-speculative-tokens is required with --drafter")
    if args.fold_window and not args.drafter:
        p.error("FOLD needs a drafter")
    return args


def setup_environment(args):
    """Must run before vLLM is imported. vLLM worker processes inherit FOLD_CONFIG."""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"  # fold.vllm_plugin wraps the V2 GPU model runner
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if "CUDA_HOME" not in os.environ:
        print("warning: CUDA_HOME is not set. vLLM JIT-compiles FlashInfer sampling kernels at startup "
              "and needs a CUDA toolkit that matches the CUDA version of PyTorch.", file=sys.stderr)
    os.environ.pop("FOLD_CONFIG", None)
    if args.fold_window:
        cfg = dict(window=args.fold_window, alpha=args.fold_alpha, mode=args.fold_mode,
                   sliding_layers=args.fold_sliding_layers, local_splits=args.fold_local_splits)
        if args.fold_mode == "rerope":
            cfg.update(alpha=1.0, logn_ref=args.fold_logn_ref)
        os.environ["FOLD_CONFIG"] = json.dumps(cfg)


def engine_args(args):
    from vllm.engine.arg_utils import EngineArgs
    kw = dict(model=args.target, tokenizer=args.target, dtype="bfloat16", tensor_parallel_size=1,
              max_num_seqs=1, max_model_len=args.max_model_len, max_num_batched_tokens=8192,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0, skip_tokenizer_init=True,
              language_model_only=True, generation_config="vllm", disable_log_stats=False,
              async_scheduling=False)
    sizes = {1, 2, 4, 8, 16}
    if args.drafter:
        n = args.num_speculative_tokens
        kw["speculative_config"] = dict(method=args.method, model=args.drafter, num_speculative_tokens=n,
                                        draft_tensor_parallel_size=1, parallel_drafting=True,
                                        enable_adaptive_verification=False)
        if args.method == "dflash":
            sizes.add(n + 1)  # the DFlash block holds the anchor and n mask positions
    compilation = dict(cudagraph_capture_sizes=sorted(sizes))
    if args.deterministic_inductor:
        compilation.update(compile_sizes=[], inductor_compile_config=dict(
            deterministic=True, combo_kernels=True, benchmark_combo_kernel=False, combo_kernels_pointwise_only=True))
    kw["compilation_config"] = compilation
    if args.drafter and args.draft_max_position:
        from fold.rope_cache import DraftRopeCache
        kw["hf_overrides"] = DraftRopeCache(args.draft_max_position)
    return EngineArgs(**kw)


def default_stop_token_ids(target):
    """eos_token_id of the target's generation config, or of its (text) config."""
    from transformers import AutoConfig, GenerationConfig
    try:
        eos = GenerationConfig.from_pretrained(target).eos_token_id
    except OSError:
        eos = None
    if eos is None:
        config = AutoConfig.from_pretrained(target)
        eos = getattr(config, "eos_token_id", None) or getattr(config.get_text_config(), "eos_token_id", None)
    if eos is None:
        raise SystemExit("no eos_token_id found for the target, pass --stop-token-ids")
    return [eos] if isinstance(eos, int) else list(eos)


def spec_counters(engine):
    values = dict.fromkeys(SPEC_COUNTERS, 0)
    for metric in engine.get_metrics():
        for name in SPEC_COUNTERS:
            if metric.name == "vllm:spec_decode_" + name:
                values[name] += int(metric.value)
    return values


def run_request(engine, request_id, input_ids, max_new_tokens, stop_token_ids):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_new_tokens,
                            stop_token_ids=stop_token_ids, ignore_eos=False, detokenize=False,
                            output_kind=RequestOutputKind.CUMULATIVE)
    before = spec_counters(engine)
    start = time.perf_counter()
    engine.add_request(request_id, {"prompt_token_ids": input_ids}, params)
    first = final = result = None
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            if out.outputs[0].token_ids and first is None:
                first = now
            if out.finished:
                final, result = now, out
    after = spec_counters(engine)
    ids = list(result.outputs[0].token_ids)
    delta = {k: after[k] - before[k] for k in SPEC_COUNTERS}
    rounds = delta["num_drafts"]
    return dict(output_ids=ids, output_tokens=len(ids), finish_reason=result.outputs[0].finish_reason,
                ttft=first - start, tpot=(final - first) / (len(ids) - 1) if len(ids) > 1 else None,
                tau=1 + delta["num_accepted_tokens"] / rounds if rounds else None,
                rounds=rounds, accepted=delta["num_accepted_tokens"])


def main():
    args = parse_args()
    setup_environment(args)
    rows = [json.loads(line) for line in args.inputs.open() if line.strip()]
    if not args.stop_token_ids:
        args.stop_token_ids = default_stop_token_ids(args.target)
        print("stop token ids:", args.stop_token_ids, flush=True)
    from vllm.v1.engine.llm_engine import LLMEngine
    engine = LLMEngine.from_engine_args(engine_args(args))
    try:
        for i in range(args.warmup_runs):
            run_request(engine, f"warmup-{i}", rows[0]["input_ids"][:args.warmup_tokens], 16, args.stop_token_ids)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for i, row in enumerate(rows):
                record = run_request(engine, f"request-{i}", row["input_ids"], row["max_new_tokens"], args.stop_token_ids)
                record = dict(id=row["id"], task=row.get("task"), prompt_tokens=len(row["input_ids"]), **record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                tau = "-" if record["tau"] is None else f"{record['tau']:.2f}"
                print(f"[{i + 1}/{len(rows)}] {row['id']} tokens={record['output_tokens']} "
                      f"tpot={record['tpot'] * 1e3 if record['tpot'] else 0:.2f}ms tau={tau}", flush=True)
    finally:
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
