from collections import deque
from enum import Enum

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class KVBlockState(str, Enum):
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    BOTH = "both"
    EMPTY = "empty"


class Block:

    def __init__(self, logical_block_id: int):
        self.logical_block_id = logical_block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.gpu_block_id: int | None = None
        self.cpu_block_id: int | None = None
        self.last_access_step = 0
        self.dirty = False

    @property
    def block_id(self) -> int:
        # Backward-compatible alias for code that still thinks in logical ids.
        return self.logical_block_id

    @property
    def state(self) -> KVBlockState:
        if self.gpu_block_id is not None and self.cpu_block_id is not None:
            return KVBlockState.BOTH
        if self.gpu_block_id is not None:
            return KVBlockState.GPU_ONLY
        if self.cpu_block_id is not None:
            return KVBlockState.CPU_ONLY
        return KVBlockState.EMPTY

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.gpu_block_id = None
        self.cpu_block_id = None
        self.last_access_step = 0
        self.dirty = False


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_kv_offload: bool = False,
        num_cpu_blocks: int = 0,
    ):
        self.num_gpu_blocks = num_blocks
        self.block_size = block_size
        self.enable_kv_offload = enable_kv_offload
        self.blocks: list[Block] = [] if enable_kv_offload else [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_gpu_block_ids: deque[int] = deque(range(num_blocks))
        self.free_cpu_block_ids: deque[int] = deque(range(num_cpu_blocks))
        self.free_logical_block_ids: deque[int] = deque() if enable_kv_offload else deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.kv_offload_manager = None
        self.access_step = 0

    @property
    def free_block_ids(self) -> deque[int]:
        # Compatibility alias: callers historically meant free GPU slots.
        return self.free_gpu_block_ids

    def bind_kv_offload_manager(self, kv_offload_manager) -> None:
        self.kv_offload_manager = kv_offload_manager

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _new_logical_block(self) -> Block:
        if self.free_logical_block_ids:
            logical_block_id = self.free_logical_block_ids.popleft()
            block = self.blocks[logical_block_id]
        else:
            logical_block_id = len(self.blocks)
            block = Block(logical_block_id)
            self.blocks.append(block)
        block.reset()
        self.used_block_ids.add(logical_block_id)
        return block

    def _allocate_gpu_block_id(self) -> int:
        if not self.free_gpu_block_ids:
            raise RuntimeError("No free GPU KV cache blocks")
        return self.free_gpu_block_ids.popleft()

    def _release_gpu_block_id(self, gpu_block_id: int) -> None:
        self.free_gpu_block_ids.append(gpu_block_id)

    def _allocate_cpu_block_id(self) -> int:
        if not self.free_cpu_block_ids:
            raise RuntimeError(
                "No free CPU KV cache blocks for offload. Increase num_cpu_kvcache_blocks or cpu_kvcache_gb."
            )
        return self.free_cpu_block_ids.popleft()

    def _release_cpu_block_id(self, cpu_block_id: int) -> None:
        self.free_cpu_block_ids.append(cpu_block_id)

    def allocate_cpu_slot(self, block: Block) -> int:
        if block.cpu_block_id is None:
            block.cpu_block_id = self._allocate_cpu_block_id()
        return block.cpu_block_id

    def allocate_gpu_slot(self, block: Block, protected_logical_block_ids: set[int] | None = None) -> int:
        if block.gpu_block_id is None:
            if not self.free_gpu_block_ids and self.enable_kv_offload:
                protected = set() if protected_logical_block_ids is None else set(protected_logical_block_ids)
                protected.add(block.logical_block_id)
                self.evict_gpu_blocks_for_space(1, protected)
            block.gpu_block_id = self._allocate_gpu_block_id()
        return block.gpu_block_id

    def release_gpu_slot(self, block: Block) -> None:
        if block.gpu_block_id is None:
            return
        self._release_gpu_block_id(block.gpu_block_id)
        block.gpu_block_id = None

    def _allocate_block(self, protected_logical_block_ids: set[int] | None = None) -> Block:
        block = self._new_logical_block()
        if self.enable_kv_offload:
            self.allocate_gpu_slot(block, protected_logical_block_ids)
        else:
            self.free_gpu_block_ids.remove(block.logical_block_id)
            block.gpu_block_id = block.logical_block_id
        block.dirty = True
        return block

    def _deallocate_block(self, logical_block_id: int) -> None:
        block = self.blocks[logical_block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == logical_block_id:
            del self.hash_to_block_id[block.hash]
        self.release_gpu_slot(block)
        if block.cpu_block_id is not None:
            self._release_cpu_block_id(block.cpu_block_id)
            block.cpu_block_id = None
        self.used_block_ids.remove(logical_block_id)
        self.free_logical_block_ids.append(logical_block_id)

    def _unique_logical_ids(self, logical_block_ids) -> list[int]:
        return list(dict.fromkeys(logical_block_ids))

    def logical_ids_from_seqs(self, seqs: list[Sequence]) -> list[int]:
        logical_block_ids = []
        for seq in seqs:
            logical_block_ids.extend(seq.block_table)
        return self._unique_logical_ids(logical_block_ids)

    def _num_blocks_needing_gpu(self, logical_block_ids) -> int:
        return sum(1 for logical_id in set(logical_block_ids) if self.blocks[logical_id].gpu_block_id is None)

    def can_ensure_blocks_on_gpu(self, logical_block_ids, protected_logical_block_ids=None) -> bool:
        if not self.enable_kv_offload:
            return len(self.free_gpu_block_ids) >= self._num_blocks_needing_gpu(logical_block_ids)
        protected = set(logical_block_ids)
        if protected_logical_block_ids is not None:
            protected.update(protected_logical_block_ids)
        return len(protected) <= self.num_gpu_blocks

    def evict_gpu_blocks_for_space(
        self,
        required_gpu_blocks: int,
        protected_logical_block_ids: set[int] | None = None,
    ) -> bool:
        if len(self.free_gpu_block_ids) >= required_gpu_blocks:
            return True
        if not self.enable_kv_offload or self.kv_offload_manager is None:
            return False
        protected = set() if protected_logical_block_ids is None else set(protected_logical_block_ids)
        while len(self.free_gpu_block_ids) < required_gpu_blocks:
            candidates = [
                self.blocks[logical_block_id]
                for logical_block_id in self.used_block_ids
                if logical_block_id not in protected and self.blocks[logical_block_id].gpu_block_id is not None
            ]
            if not candidates:
                return False
            victim = min(candidates, key=lambda block: block.last_access_step)
            self.kv_offload_manager.offload_block(victim)
            self.release_gpu_slot(victim)
        return True

    def ensure_free_gpu_blocks(self, required_gpu_blocks: int, protected_logical_block_ids=None) -> bool:
        if len(self.free_gpu_block_ids) >= required_gpu_blocks:
            return True
        if not self.enable_kv_offload:
            return False
        protected = set() if protected_logical_block_ids is None else set(protected_logical_block_ids)
        return self.evict_gpu_blocks_for_space(required_gpu_blocks, protected)

    def touch_blocks(self, logical_block_ids) -> None:
        for logical_block_id in self._unique_logical_ids(logical_block_ids):
            self.access_step += 1
            self.blocks[logical_block_id].last_access_step = self.access_step

    def ensure_blocks_on_gpu(self, logical_block_ids, protected_logical_block_ids=None) -> None:
        logical_block_ids = self._unique_logical_ids(logical_block_ids)
        if not logical_block_ids:
            return
        protected = set(logical_block_ids)
        if protected_logical_block_ids is not None:
            protected.update(protected_logical_block_ids)
        if len(protected) > self.num_gpu_blocks:
            raise RuntimeError(
                f"KV working set needs {len(protected)} GPU blocks, but only {self.num_gpu_blocks} are available"
            )
        missing = self._num_blocks_needing_gpu(logical_block_ids)
        if self.kv_offload_manager is None:
            if len(self.free_gpu_block_ids) < missing:
                raise RuntimeError("KV offload manager is not bound and GPU KV blocks are not resident")
            self.touch_blocks(logical_block_ids)
            return
        self.kv_offload_manager.ensure_blocks_on_gpu(logical_block_ids, protected)
        self.touch_blocks(logical_block_ids)

    def build_gpu_block_table(self, seq: Sequence, protected_logical_block_ids=None) -> list[int]:
        self.ensure_blocks_on_gpu(seq.block_table, protected_logical_block_ids)
        gpu_block_table = []
        for logical_block_id in seq.block_table:
            gpu_block_id = self.blocks[logical_block_id].gpu_block_id
            if gpu_block_id is None:
                raise RuntimeError(f"Logical KV block {logical_block_id} is not resident on GPU")
            gpu_block_table.append(gpu_block_id)
        return gpu_block_table

    def can_allocate(self, seq: Sequence, protected_logical_block_ids=None) -> bool:
        protected = set() if protected_logical_block_ids is None else set(protected_logical_block_ids)
        if seq.block_table:
            return self.can_ensure_blocks_on_gpu(seq.block_table[:seq.num_prefill_blocks], protected)
        if not self.enable_kv_offload:
            return len(self.free_gpu_block_ids) >= seq.num_prefill_blocks
        return len(protected) + seq.num_prefill_blocks <= self.num_gpu_blocks

    def allocate(self, seq: Sequence, protected_logical_block_ids=None):
        protected = set() if protected_logical_block_ids is None else set(protected_logical_block_ids)
        if seq.block_table:
            self.ensure_blocks_on_gpu(seq.block_table[:seq.num_prefill_blocks], protected)
            return
        if not self.can_allocate(seq, protected):
            raise RuntimeError("Sequence prefill working set does not fit in GPU KV cache")
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0
        h = -1
        cache_miss = False
        for i in range(seq.num_prefill_blocks):
            token_ids = seq.prefill_block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            logical_block_id = self.hash_to_block_id.get(h, -1)
            if logical_block_id == -1 or self.blocks[logical_block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block = self._allocate_block(protected | set(seq.block_table))
                logical_block_id = block.logical_block_id
            else:
                seq.num_cached_tokens += self.block_size
                block = self.blocks[logical_block_id]
                block.ref_count += 1
                self.ensure_blocks_on_gpu([logical_block_id], protected | set(seq.block_table))
            if h != -1 and not cache_miss:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = logical_block_id
            seq.block_table.append(logical_block_id)
        seq.num_computed_tokens = seq.num_cached_tokens
        self.ensure_blocks_on_gpu(seq.block_table[:seq.num_prefill_blocks], protected | set(seq.block_table))

    def can_append(self, seq: Sequence, protected_logical_block_ids=None) -> bool:
        if len(seq) % self.block_size != 1:
            return True
        protected = set(seq.block_table)
        if protected_logical_block_ids is not None:
            protected.update(protected_logical_block_ids)
        if not self.enable_kv_offload:
            return len(self.free_gpu_block_ids) >= 1
        return len(protected) + 1 <= self.num_gpu_blocks

    def may_append(self, seq: Sequence, protected_logical_block_ids=None):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        protected = set(block_table)
        if protected_logical_block_ids is not None:
            protected.update(protected_logical_block_ids)
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            if not self.ensure_free_gpu_blocks(1, protected):
                raise RuntimeError("No GPU KV block can be freed for decode append")
            block = self._allocate_block(protected)
            block_table.append(block.logical_block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            last_block.dirty = True
            self.hash_to_block_id[h] = last_block.logical_block_id
        else:
            assert last_block.hash == -1
            last_block.dirty = True

    def deallocate(self, seq: Sequence):
        for logical_block_id in reversed(seq.block_table):
            block = self.blocks[logical_block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(logical_block_id)
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0
        seq.scheduled_prefill_end = 0
        seq.sample_after_prefill = False
        seq.block_table.clear()

    def may_prefill(self, seq: Sequence, prefill_end: int):
        start_block = seq.num_computed_tokens // self.block_size
        end_block = prefill_end // self.block_size
        for i in range(start_block, end_block):
            block = self.blocks[seq.block_table[i]]
            block.dirty = True
            if block.hash != -1:
                continue
            token_ids = seq.prefill_block(i)
            prefix = self.blocks[seq.block_table[i-1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.logical_block_id
