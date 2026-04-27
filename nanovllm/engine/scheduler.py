from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_kv_offload = config.enable_kv_offload
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_kv_offload,
            config.num_cpu_kvcache_blocks or 0,
        )
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.last_decode_batch: list[Sequence] = []

    def is_finished(self):
        return not self.waiting and not self.prefilling and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _should_finish(self, seq: Sequence, token_id: int):
        return (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens

    def _schedule_prefill_chunk(self, seq: Sequence, num_batched_tokens: int) -> int:
        remaining_budget = self.max_num_batched_tokens - num_batched_tokens
        num_prefill_tokens = seq.num_prefill_tokens - seq.num_computed_tokens
        assert remaining_budget > 0 and num_prefill_tokens > 0
        num_chunk_tokens = min(num_prefill_tokens, remaining_budget)
        seq.scheduled_prefill_end = seq.num_computed_tokens + num_chunk_tokens
        seq.sample_after_prefill = seq.num_completion_tokens == 0 and seq.scheduled_prefill_end == seq.num_prompt_tokens
        return num_chunk_tokens

    def schedule_prefill(self, reserved_tokens: int = 0) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = reserved_tokens
        protected = set(self.block_manager.logical_ids_from_seqs(self.last_decode_batch))
        while self.prefilling and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.prefilling.popleft()
            if not self.block_manager.can_allocate(seq, protected):
                self.prefilling.appendleft(seq)
                break
            self.block_manager.allocate(seq, protected)
            num_batched_tokens += self._schedule_prefill_chunk(seq, num_batched_tokens)
            scheduled_seqs.append(seq)
            protected.update(seq.block_table)
        num_seqs = len(self.running) + len(self.prefilling) + len(scheduled_seqs)
        while self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq, protected):
                break
            num_seqs += 1
            self.block_manager.allocate(seq, protected)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            if seq.num_prefill_tokens == seq.num_computed_tokens:
                if seq.num_completion_tokens == 0:
                    seq.num_computed_tokens -= 1
                else:
                    self.running.append(seq)
                    protected.update(seq.block_table)
                    continue
            num_batched_tokens += self._schedule_prefill_chunk(seq, num_batched_tokens)
            scheduled_seqs.append(seq)
            protected.update(seq.block_table)
        return scheduled_seqs

    def schedule_decode(self, token_budget: int | None = None) -> list[Sequence]:
        scheduled_seqs = []
        skipped_seqs = []
        protected = set()
        num_seqs = 0
        num_tokens = 0
        token_budget = self.max_num_batched_tokens if token_budget is None else token_budget
        while self.running and num_seqs < self.max_num_seqs and num_tokens < token_budget:
            seq = self.running.popleft()
            candidate_protected = protected | set(seq.block_table)
            if not self.block_manager.can_ensure_blocks_on_gpu(seq.block_table, candidate_protected):
                skipped_seqs.append(seq)
                continue
            if not self.block_manager.can_append(seq, candidate_protected):
                skipped_seqs.append(seq)
                continue
            num_seqs += 1
            num_tokens += 1
            self.block_manager.may_append(seq, candidate_protected)
            protected.update(seq.block_table)
            scheduled_seqs.append(seq)
        self.last_decode_batch = scheduled_seqs
        self.running.extendleft(reversed(skipped_seqs))
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def postprocess_prefill(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.may_prefill(seq, seq.scheduled_prefill_end)
            seq.num_computed_tokens = seq.scheduled_prefill_end
            seq.scheduled_prefill_end = 0
            if seq.sample_after_prefill:
                seq.sample_after_prefill = False
                seq.append_token(token_id)
                if self._should_finish(seq, token_id):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                else:
                    self.running.append(seq)
            elif seq.needs_prefill:
                seq.sample_after_prefill = False
                self.prefilling.append(seq)
            else:
                seq.sample_after_prefill = False
                self.running.append(seq)

    def postprocess_decode(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens += 1
            seq.append_token(token_id)
            if self._should_finish(seq, token_id):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
