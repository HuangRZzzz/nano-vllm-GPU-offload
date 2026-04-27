import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.enable_mixed_batch = config.enable_mixed_batch
        self.enable_kv_offload = config.enable_kv_offload
        self.last_step_was_mixed = False
        self.scheduler = Scheduler(config)
        self.model_runner.bind_block_manager(self.scheduler.block_manager)
        atexit.register(self.exit)

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return
        model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _run_prefill(self, seqs: list[Sequence]):
        if not seqs:
            return [], 0, 0.
        num_tokens = sum(seq.scheduled_prefill_end - seq.num_computed_tokens for seq in seqs)
        t = perf_counter()
        token_ids = self.model_runner.call("run", seqs, True)
        self.scheduler.postprocess_prefill(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, perf_counter() - t

    def _run_decode(self, seqs: list[Sequence]):
        if not seqs:
            return [], 0, 0.
        t = perf_counter()
        token_ids = self.model_runner.call("run", seqs, False)
        self.scheduler.postprocess_decode(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, len(seqs), perf_counter() - t

    def _run_mixed(self, decode_seqs: list[Sequence], prefill_seqs: list[Sequence]):
        num_decode_tokens = len(decode_seqs)
        num_prefill_tokens = sum(seq.scheduled_prefill_end - seq.num_computed_tokens for seq in prefill_seqs)
        t = perf_counter()
        token_ids = self.model_runner.call("run_mixed", decode_seqs, prefill_seqs)
        mixed_time = perf_counter() - t
        decode_token_ids = token_ids[:len(decode_seqs)]
        prefill_token_ids = token_ids[len(decode_seqs):]
        self.scheduler.postprocess_decode(decode_seqs, decode_token_ids)
        self.scheduler.postprocess_prefill(prefill_seqs, prefill_token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in decode_seqs if seq.is_finished]
        outputs.extend((seq.seq_id, seq.completion_token_ids) for seq in prefill_seqs if seq.is_finished)
        return outputs, num_prefill_tokens, mixed_time, num_decode_tokens, mixed_time

    def step(self):
        outputs = []
        self.last_step_was_mixed = False
        decode_seqs = self.scheduler.schedule_decode()
        prefill_reserved_tokens = len(decode_seqs) if self.enable_mixed_batch else 0
        prefill_seqs = self.scheduler.schedule_prefill(reserved_tokens=prefill_reserved_tokens)
        if self.enable_mixed_batch and decode_seqs and prefill_seqs:
            self.last_step_was_mixed = True
            mixed_outputs, num_prefill_tokens, prefill_time, num_decode_tokens, decode_time = self._run_mixed(
                decode_seqs, prefill_seqs)
            outputs.extend(mixed_outputs)
        else:
            decode_outputs, num_decode_tokens, decode_time = self._run_decode(decode_seqs)
            outputs.extend(decode_outputs)
            prefill_outputs, num_prefill_tokens, prefill_time = self._run_prefill(prefill_seqs)
            outputs.extend(prefill_outputs)
        if not decode_seqs and not prefill_seqs and not self.is_finished():
            raise RuntimeError("scheduler made no progress; request may exceed KV cache capacity")
        return outputs, num_prefill_tokens, prefill_time, num_decode_tokens, decode_time

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            output, num_prefill_tokens, prefill_time, num_decode_tokens, decode_time = self.step()
            if use_tqdm:
                if num_prefill_tokens > 0 and prefill_time > 0:
                    prefill_throughput = num_prefill_tokens / prefill_time
                if num_decode_tokens > 0 and decode_time > 0:
                    decode_throughput = num_decode_tokens / decode_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
