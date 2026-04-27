import os
from nanovllm import LLM, SamplingParams
from nanovllm.engine.kv_offload import KVOffloadManager
from transformers import AutoTokenizer


def trace_kv_offload():
    original_offload = KVOffloadManager.offload_block
    original_load = KVOffloadManager.load_block_to_gpu

    def offload_block(self, block):
        print(
            f"[kv-offload] D2H logical={block.logical_block_id} "
            f"gpu={block.gpu_block_id} cpu={block.cpu_block_id}"
        )
        return original_offload(self, block)

    def load_block_to_gpu(self, block, protected_logical_block_ids=None):
        if block.gpu_block_id is None:
            print(
                f"[kv-offload] H2D logical={block.logical_block_id} "
                f"gpu=<allocating> cpu={block.cpu_block_id}"
            )
        return original_load(self, block, protected_logical_block_ids)

    KVOffloadManager.offload_block = offload_block
    KVOffloadManager.load_block_to_gpu = load_block_to_gpu


def make_prompt(tokenizer, target_len: int) -> list[int]:
    token_ids = tokenizer.encode("你是千问大模型吗？请继续回答。", add_special_tokens=False)
    repeats = (target_len + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:target_len]


def main():
    path = os.environ.get("MODEL_PATH", "/home/zl/hrz/temp/nano-vllm/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    trace_kv_offload()
    llm = LLM(
        path,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_kv_offload=True,
        num_cpu_kvcache_blocks=8,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        max_model_len=512,
        max_num_kvcache_blocks=2,
        min_num_kvcache_blocks=1,
        min_kvcache_memory_bytes=0,
    )

    sampling_params = SamplingParams(temperature=1, max_tokens=4, ignore_eos=True)
    prompts = [make_prompt(tokenizer, 255), make_prompt(tokenizer, 255)]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt length: {len(prompt)} tokens")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
