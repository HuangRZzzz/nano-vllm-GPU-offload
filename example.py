import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = r"/home/zl/hrz/dir"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        tensor_parallel_size=1,
        cpu_offload_num_layers="auto    ",
        cpu_offload_window_size=1,
    )

    sampling_params = SamplingParams(temperature=1, max_tokens=256)
    prompts = [
        "你是千问大模型吗",
        "你是千问大模型吗",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
