import argparse
import atexit
import gc
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

from nanovllm import LLM, SamplingParams


@dataclass
class RequestSpec:
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float


@dataclass
class RequestState:
    seq: object
    prompt_tokens: int
    target_output_tokens: int
    planned_arrival_time: float
    submit_time: float
    first_token_time: float | None = None
    finish_time: float | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark nano-vllm with burst or stream arrivals and report "
            "prefill/decode throughput, TTFT, and request latency."
        )
    )
    parser.add_argument("--model-path", type=str, default="/home/zl/hrz/temp/nano-vllm/Qwen3-0.6B")
    parser.add_argument("--mode", choices=("burst", "stream"), default="burst")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--qps", type=float, default=32.0)
    parser.add_argument("--fixed-prompt-len", type=int, default=None)
    parser.add_argument("--fixed-output-len", type=int, default=None)
    parser.add_argument("--min-prompt-len", type=int, default=100)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--prompt-lens", type=str, default=None)
    parser.add_argument("--output-lens", type=str, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--min-num-kvcache-blocks", type=int, default=None)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=None)
    parser.add_argument("--min-kvcache-memory-bytes", type=int, default=1024**3)
    parser.add_argument("--cpu-offload-num-layers", type=str, default="0")
    parser.add_argument("--cpu-offload-window-size", type=int, default=1)
    parser.add_argument("--kv-offload", choices=("on", "off"), default="off")
    parser.add_argument("--num-cpu-kvcache-blocks", type=int, default=None)
    parser.add_argument("--cpu-kvcache-gb", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--mixed-batch", choices=("on", "off", "compare"), default="on")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def resolve_model_path(cli_path: str | None) -> str:
    candidates = []
    if cli_path:
        candidates.append(Path(os.path.expanduser(cli_path)))
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidates.append(Path(os.path.expanduser(env_path)))
    candidates.extend([
        Path(__file__).resolve().parent / "Qwen3-0.6B",
        Path(os.path.expanduser("~/huggingface/Qwen3-0.6B")),
    ])

    weight_files = ("model.safetensors", "model.safetensors.index.json")
    checked = []
    for path in candidates:
        checked.append(str(path))
        if path.is_dir() and any((path / name).exists() for name in weight_files):
            return str(path)
    raise FileNotFoundError(
        "Could not find complete model weights. "
        f"Checked: {', '.join(checked)}"
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def sample_length(rng: Random, fixed_len: int | None, min_len: int, max_len: int) -> int:
    if fixed_len is not None:
        return fixed_len
    return rng.randint(min_len, max_len)


def build_request_specs(
    args,
    prompt_len: int | None,
    output_len: int | None,
    seed: int,
) -> list[RequestSpec]:
    rng = Random(seed)
    specs = []
    for i in range(args.num_seqs):
        prompt_tokens = sample_length(rng, prompt_len, args.min_prompt_len, args.max_prompt_len)
        output_tokens = sample_length(rng, output_len, args.min_output_len, args.max_output_len)
        arrival_time = 0.0 if args.mode == "burst" else i / args.qps
        prompt_token_ids = [rng.randint(0, 10000) for _ in range(prompt_tokens)]
        specs.append(RequestSpec(prompt_token_ids, output_tokens, arrival_time))
    return specs


def warmup_llm(llm: LLM, max_model_len: int):
    warmup_prompt = [0] * min(32, max_model_len)
    llm.generate(
        [warmup_prompt],
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4),
        use_tqdm=False,
    )


def create_llm(args, model_path: str, enable_mixed_batch: bool) -> LLM:
    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        enable_mixed_batch=enable_mixed_batch,
        enable_kv_offload=args.kv_offload == "on",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        min_num_kvcache_blocks=args.min_num_kvcache_blocks,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        min_kvcache_memory_bytes=args.min_kvcache_memory_bytes,
        num_cpu_kvcache_blocks=args.num_cpu_kvcache_blocks,
        cpu_kvcache_gb=args.cpu_kvcache_gb,
        cpu_offload_num_layers=args.cpu_offload_num_layers,
        cpu_offload_window_size=args.cpu_offload_window_size,
    )
    warmup_llm(llm, args.max_model_len)
    return llm


def close_llm(llm: LLM):
    atexit.unregister(llm.exit)
    llm.exit()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def collect_kv_offload_stats(llm: LLM):
    manager = getattr(llm.model_runner, "kv_offload_manager", None)
    stats = {"d2h": 0, "h2d": 0}
    if manager is None:
        return stats, lambda: None

    original_offload = manager.offload_block
    original_load = manager.load_block_to_gpu

    def offload_block(block):
        stats["d2h"] += 1
        return original_offload(block)

    def load_block_to_gpu(block, protected_logical_block_ids=None):
        if block.gpu_block_id is None:
            stats["h2d"] += 1
        return original_load(block, protected_logical_block_ids)

    manager.offload_block = offload_block
    manager.load_block_to_gpu = load_block_to_gpu

    def restore():
        manager.offload_block = original_offload
        manager.load_block_to_gpu = original_load

    return stats, restore


def run_once(
    args,
    model_path: str,
    prompt_len: int | None,
    output_len: int | None,
    seed: int,
    enable_mixed_batch: bool,
):
    llm = create_llm(args, model_path, enable_mixed_batch)
    kv_offload_stats, restore_kv_offload_stats = collect_kv_offload_stats(llm)
    try:
        specs = build_request_specs(args, prompt_len, output_len, seed)
        states: list[RequestState] = []
        next_submit_idx = 0
        total_prompt_tokens = sum(len(spec.prompt_token_ids) for spec in specs)
        total_prefill_tokens = 0
        total_decode_tokens = 0
        total_prefill_time = 0.0
        total_decode_time = 0.0
        total_busy_time = 0.0
        mixed_time = 0.0
        mixed_steps = 0
        start = time.perf_counter()

        while next_submit_idx < len(specs) or not llm.is_finished():
            now = time.perf_counter() - start
            while next_submit_idx < len(specs) and specs[next_submit_idx].arrival_time <= now:
                spec = specs[next_submit_idx]
                llm.add_request(
                    spec.prompt_token_ids,
                    SamplingParams(
                        temperature=args.temperature,
                        ignore_eos=True,
                        max_tokens=spec.max_tokens,
                    ),
                )
                seq = llm.scheduler.waiting[-1]
                states.append(
                    RequestState(
                        seq=seq,
                        prompt_tokens=len(spec.prompt_token_ids),
                        target_output_tokens=spec.max_tokens,
                        planned_arrival_time=spec.arrival_time,
                        submit_time=now,
                    )
                )
                next_submit_idx += 1

            if llm.is_finished():
                if next_submit_idx >= len(specs):
                    break
                sleep_for = specs[next_submit_idx].arrival_time - (time.perf_counter() - start)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.01))
                continue

            step_start = time.perf_counter()
            step_result = llm.step()
            step_time = time.perf_counter() - step_start
            if isinstance(step_result, tuple) and len(step_result) == 5:
                _, num_prefill_tokens, prefill_time, num_decode_tokens, decode_time = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 2:
                _, num_tokens = step_result
                if num_tokens >= 0:
                    num_prefill_tokens, prefill_time = num_tokens, step_time
                    num_decode_tokens, decode_time = 0, 0.0
                else:
                    num_prefill_tokens, prefill_time = 0, 0.0
                    num_decode_tokens, decode_time = -num_tokens, step_time
            else:
                raise RuntimeError(f"Unsupported step() return format: {type(step_result)!r}")
            total_busy_time += step_time
            total_prefill_tokens += num_prefill_tokens
            total_prefill_time += prefill_time
            total_decode_tokens += num_decode_tokens
            total_decode_time += decode_time
            if llm.last_step_was_mixed:
                mixed_steps += 1
                mixed_time += step_time
            now = time.perf_counter() - start

            for state in states:
                if state.first_token_time is None and state.seq.num_completion_tokens > 0:
                    state.first_token_time = now
                if state.finish_time is None and state.seq.is_finished:
                    state.finish_time = now

        wall_time = time.perf_counter() - start
        ttfts = [state.first_token_time - state.submit_time for state in states if state.first_token_time is not None]
        latencies = [state.finish_time - state.submit_time for state in states if state.finish_time is not None]
        submit_delays = [state.submit_time - state.planned_arrival_time for state in states]
        total_output_tokens = sum(state.seq.num_completion_tokens for state in states)
        total_tokens = total_prompt_tokens + total_output_tokens
        busy_time = total_busy_time

        return {
            "seed": seed,
            "mixed_batch_enabled": enable_mixed_batch,
            "num_requests": len(states),
            "wall_time_s": wall_time,
            "engine_busy_time_s": busy_time,
            "gpu_busy_pct": 100.0 * busy_time / wall_time if wall_time > 0 else 0.0,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "prefill_tokens": total_prefill_tokens,
            "decode_tokens": total_decode_tokens,
            "prefill_time_s": total_prefill_time,
            "decode_time_s": total_decode_time,
            "mixed_time_s": mixed_time,
            "mixed_steps": mixed_steps,
            "request_throughput_rps": len(states) / wall_time if wall_time > 0 else 0.0,
            "output_throughput_tok_s": total_output_tokens / wall_time if wall_time > 0 else 0.0,
            "total_throughput_tok_s": total_tokens / wall_time if wall_time > 0 else 0.0,
            "prefill_throughput_tok_s": total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0.0,
            "decode_throughput_tok_s": total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0,
            "kv_d2h": kv_offload_stats["d2h"],
            "kv_h2d": kv_offload_stats["h2d"],
            "ttft_p50_s": percentile(ttfts, 50),
            "ttft_p95_s": percentile(ttfts, 95),
            "ttft_p99_s": percentile(ttfts, 99),
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "latency_p99_s": percentile(latencies, 99),
            "submit_delay_p50_s": percentile(submit_delays, 50),
            "submit_delay_p95_s": percentile(submit_delays, 95),
        }
    finally:
        restore_kv_offload_stats()
        close_llm(llm)


def summarize_runs(runs: list[dict]) -> dict:
    keys = [
        "wall_time_s",
        "engine_busy_time_s",
        "gpu_busy_pct",
        "request_throughput_rps",
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "prefill_throughput_tok_s",
        "decode_throughput_tok_s",
        "kv_d2h",
        "kv_h2d",
        "ttft_p50_s",
        "ttft_p95_s",
        "latency_p50_s",
        "latency_p95_s",
        "submit_delay_p50_s",
        "submit_delay_p95_s",
    ]
    summary = {}
    for key in keys:
        values = [run[key] for run in runs]
        mean, std = mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    summary["prompt_tokens_mean"] = statistics.mean(run["prompt_tokens"] for run in runs)
    summary["output_tokens_mean"] = statistics.mean(run["output_tokens"] for run in runs)
    summary["total_tokens_mean"] = statistics.mean(run["total_tokens"] for run in runs)
    summary["num_requests_mean"] = statistics.mean(run["num_requests"] for run in runs)
    summary["mixed_time_s_mean"] = statistics.mean(run["mixed_time_s"] for run in runs)
    summary["mixed_steps_mean"] = statistics.mean(run["mixed_steps"] for run in runs)
    return summary


def case_label(args, prompt_len: int | None, output_len: int | None, enable_mixed_batch: bool) -> str:
    prompt_desc = (
        str(prompt_len)
        if prompt_len is not None
        else f"{args.min_prompt_len}-{args.max_prompt_len}"
    )
    output_desc = (
        str(output_len)
        if output_len is not None
        else f"{args.min_output_len}-{args.max_output_len}"
    )
    label = (
        f"mode={args.mode} num_seqs={args.num_seqs} "
        f"prompt={prompt_desc} output={output_desc} "
        f"kv_offload={args.kv_offload} "
        f"mixed_batch={'on' if enable_mixed_batch else 'off'}"
    )
    if args.mode == "stream":
        label += f" qps={args.qps}"
    return label


def print_run_summary(run_idx: int, run: dict):
    print(
        f"  run {run_idx}: wall={run['wall_time_s']:.2f}s "
        f"busy={run['engine_busy_time_s']:.2f}s "
        f"busy_pct={run['gpu_busy_pct']:.1f}% "
        f"mixed={run['mixed_time_s']:.2f}s/{run['mixed_steps']}steps "
        f"kv_d2h={run['kv_d2h']} "
        f"kv_h2d={run['kv_h2d']} "
        f"total={run['total_throughput_tok_s']:.2f}tok/s "
        f"out={run['output_throughput_tok_s']:.2f}tok/s "
        f"prefill={run['prefill_throughput_tok_s']:.2f}tok/s "
        f"decode={run['decode_throughput_tok_s']:.2f}tok/s "
        f"ttft_p50={run['ttft_p50_s']:.3f}s "
        f"ttft_p95={run['ttft_p95_s']:.3f}s "
        f"lat_p50={run['latency_p50_s']:.3f}s "
        f"lat_p95={run['latency_p95_s']:.3f}s"
    )


def print_case_summary(summary: dict):
    print(
        "  summary: "
        f"mixed={summary['mixed_time_s_mean']:.2f}s/{summary['mixed_steps_mean']:.1f}steps "
        f"kv_d2h={summary['kv_d2h_mean']:.1f} "
        f"kv_h2d={summary['kv_h2d_mean']:.1f} "
        f"total={summary['total_throughput_tok_s_mean']:.2f}±{summary['total_throughput_tok_s_std']:.2f}tok/s "
        f"out={summary['output_throughput_tok_s_mean']:.2f}±{summary['output_throughput_tok_s_std']:.2f}tok/s "
        f"prefill={summary['prefill_throughput_tok_s_mean']:.2f}±{summary['prefill_throughput_tok_s_std']:.2f}tok/s "
        f"decode={summary['decode_throughput_tok_s_mean']:.2f}±{summary['decode_throughput_tok_s_std']:.2f}tok/s "
        f"ttft_p50={summary['ttft_p50_s_mean']:.3f}s "
        f"ttft_p95={summary['ttft_p95_s_mean']:.3f}s "
        f"lat_p50={summary['latency_p50_s_mean']:.3f}s "
        f"lat_p95={summary['latency_p95_s_mean']:.3f}s"
    )


def mixed_batch_variants(args) -> list[bool]:
    if args.mixed_batch == "compare":
        return [False, True]
    return [args.mixed_batch == "on"]


def print_comparison(base_summary: dict, mixed_summary: dict):
    def pct(new: float, old: float) -> float:
        if old == 0:
            return 0.0
        return 100.0 * (new - old) / old

    print(
        "  compare: "
        f"total={pct(mixed_summary['total_throughput_tok_s_mean'], base_summary['total_throughput_tok_s_mean']):+.1f}% "
        f"out={pct(mixed_summary['output_throughput_tok_s_mean'], base_summary['output_throughput_tok_s_mean']):+.1f}% "
        f"ttft_p50={pct(mixed_summary['ttft_p50_s_mean'], base_summary['ttft_p50_s_mean']):+.1f}% "
        f"lat_p50={pct(mixed_summary['latency_p50_s_mean'], base_summary['latency_p50_s_mean']):+.1f}%"
    )


def main():
    args = parse_args()
    if args.mode == "stream" and args.qps <= 0:
        raise ValueError("--qps must be > 0 in stream mode")
    if args.fixed_prompt_len is not None and args.fixed_prompt_len <= 0:
        raise ValueError("--fixed-prompt-len must be > 0")
    if args.fixed_output_len is not None and args.fixed_output_len <= 0:
        raise ValueError("--fixed-output-len must be > 0")
    if args.min_prompt_len <= 0 or args.max_prompt_len < args.min_prompt_len:
        raise ValueError("prompt length range is invalid")
    if args.min_output_len <= 0 or args.max_output_len < args.min_output_len:
        raise ValueError("output length range is invalid")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    prompt_lens = parse_csv_ints(args.prompt_lens) or [args.fixed_prompt_len]
    output_lens = parse_csv_ints(args.output_lens) or [args.fixed_output_len]
    model_path = resolve_model_path(args.model_path)
    json_results = []

    for prompt_len in prompt_lens:
        for output_len in output_lens:
            summaries_by_variant = {}
            for enable_mixed_batch in mixed_batch_variants(args):
                print(case_label(args, prompt_len, output_len, enable_mixed_batch))
                runs = []
                for repeat_idx in range(args.repeats):
                    seed = args.seed + repeat_idx * args.seed_step
                    run = run_once(args, model_path, prompt_len, output_len, seed, enable_mixed_batch)
                    runs.append(run)
                    print_run_summary(repeat_idx + 1, run)
                summary = summarize_runs(runs)
                print_case_summary(summary)
                summaries_by_variant[enable_mixed_batch] = summary
                json_results.append(
                    {
                        "case": {
                            "mode": args.mode,
                            "num_seqs": args.num_seqs,
                            "prompt_len": prompt_len,
                            "output_len": output_len,
                            "qps": args.qps if args.mode == "stream" else None,
                            "repeats": args.repeats,
                            "kv_offload": args.kv_offload == "on",
                            "num_cpu_kvcache_blocks": args.num_cpu_kvcache_blocks,
                            "cpu_kvcache_gb": args.cpu_kvcache_gb,
                            "mixed_batch": enable_mixed_batch,
                        },
                        "runs": runs,
                        "summary": summary,
                    }
                )
            if args.mixed_batch == "compare":
                print_comparison(summaries_by_variant[False], summaries_by_variant[True])

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
