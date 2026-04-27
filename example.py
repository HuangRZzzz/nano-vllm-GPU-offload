import os
from time import perf_counter

from nanovllm import LLM, SamplingParams
from nanovllm.engine.kv_offload import KVOffloadManager
from transformers import AutoTokenizer


def trace_kv_offload():
    original_offload = KVOffloadManager.offload_block
    original_load = KVOffloadManager.load_block_to_gpu
    stats = {"d2h": 0, "h2d": 0}

    def offload_block(self, block):
        stats["d2h"] += 1
        print(
            f"[kv-offload] D2H logical={block.logical_block_id} "
            f"gpu={block.gpu_block_id} cpu={block.cpu_block_id}",
            flush=True,
        )
        result = original_offload(self, block)
        print(
            f"[kv-offload] D2H done logical={block.logical_block_id} "
            f"gpu={block.gpu_block_id} cpu={block.cpu_block_id}",
            flush=True,
        )
        return result

    def load_block_to_gpu(self, block, protected_logical_block_ids=None):
        needs_load = block.gpu_block_id is None
        if needs_load:
            stats["h2d"] += 1
            print(
                f"[kv-offload] H2D logical={block.logical_block_id} "
                f"gpu=<allocating> cpu={block.cpu_block_id}",
                flush=True,
            )
        result = original_load(self, block, protected_logical_block_ids)
        if needs_load:
            print(
                f"[kv-offload] H2D done logical={block.logical_block_id} "
                f"gpu={block.gpu_block_id} cpu={block.cpu_block_id}",
                flush=True,
            )
        return result

    KVOffloadManager.offload_block = offload_block
    KVOffloadManager.load_block_to_gpu = load_block_to_gpu
    return stats


def collect_engine_speed(llm):
    original_step = llm.step
    stats = {
        "prefill_tokens": 0,
        "prefill_time": 0.0,
        "decode_tokens": 0,
        "decode_time": 0.0,
    }

    def step_with_stats():
        result = original_step()
        _, num_prefill_tokens, prefill_time, num_decode_tokens, decode_time = result
        stats["prefill_tokens"] += num_prefill_tokens
        stats["prefill_time"] += prefill_time
        stats["decode_tokens"] += num_decode_tokens
        stats["decode_time"] += decode_time
        return result

    llm.step = step_with_stats
    return stats


def format_speed(tokens: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{tokens / seconds:.2f} tok/s"


def main():
    path = os.environ.get("MODEL_PATH", "/home/zl/hrz/temp/nano-vllm/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    block_size = 256
    kv_trace_stats = trace_kv_offload()
    llm = LLM(
        path,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_kv_offload=True,
        num_cpu_kvcache_blocks=16,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        max_model_len=768,
        max_num_kvcache_blocks=10,
        min_num_kvcache_blocks=1,
        min_kvcache_memory_bytes=0,
    )
    speed_stats = collect_engine_speed(llm)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=380)
    context = (
        "儿童图书馆里有一个温和的机器人管理员，它会在雨天点亮阅读灯，"
        "记录孩子们喜欢的书，也会把难词写成小卡片。今天两个孩子来到馆里："
        "一个想找恐龙和火山的故事，另一个想找星星和宇宙飞船的书。"
        "机器人决定把两类书放在同一张桌子上，引导他们发现古老地球和遥远宇宙之间的联系。"
        "桌上还有放大镜、彩色便签、一本地球年代表和一张星空地图。机器人告诉他们，"
        "每一页都可以慢慢读，每一个问题都值得被认真听见。雨声落在窗外，管理员阿姨把热水杯放在书车旁，"
        "小朋友们脱下雨衣，鞋尖还带着一点水光。机器人把书脊排成弧形，又把写着“火山”“陨石”“化石”“星云”的卡片摆成一条小路，"
        "请他们沿着这条小路寻找答案。"
    )
    prompts = [
        context + "请写一个很长的儿童故事开头，至少写十二个自然段，语气温暖，细节丰富，先不要结尾。",
        context + "请继续写一个很长的儿童故事，至少写十二个自然段，重点写机器人如何帮助两个孩子，先不要结尾。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    for i, prompt in enumerate(prompts):
        token_count = len(tokenizer.encode(prompt))
        block_count = (token_count + block_size - 1) // block_size
        print(f"[example] prompt={i} tokens={token_count} blocks={block_count}")
    # print(f"[example] GPU KV blocks capped at {max_gpu_kv_blocks}.")

    start_time = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed_time = perf_counter() - start_time
    completion_tokens = sum(len(output["token_ids"]) for output in outputs)
    print(
        "[kv-offload] summary "
        f"D2H={kv_trace_stats['d2h']} H2D={kv_trace_stats['h2d']}",
        flush=True,
    )
    print(
        "[example] token speed "
        f"prefill={format_speed(speed_stats['prefill_tokens'], speed_stats['prefill_time'])} "
        f"decode={format_speed(speed_stats['decode_tokens'], speed_stats['decode_time'])} "
        f"output_wall={format_speed(completion_tokens, elapsed_time)} "
        f"completion_tokens={completion_tokens} elapsed={elapsed_time:.2f}s",
        flush=True,
    )

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt tokens: {len(tokenizer.encode(prompt))}")
        print(f"Completion tokens: {len(output['token_ids'])}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
