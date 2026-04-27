import torch

from nanovllm.engine.sequence import Sequence


class KVOffloadManager:

    def __init__(self, kv_cache: torch.Tensor, cpu_kv_cache: torch.Tensor):
        self.kv_cache = kv_cache
        self.cpu_kv_cache = cpu_kv_cache
        self.block_manager = None

    def bind_block_manager(self, block_manager) -> None:
        self.block_manager = block_manager

    def _require_block_manager(self):
        if self.block_manager is None:
            raise RuntimeError("KVOffloadManager is not bound to a BlockManager")
        return self.block_manager

    def offload_block(self, block) -> None:
        block_manager = self._require_block_manager()
        if block.gpu_block_id is None:
            return
        cpu_block_id = block_manager.allocate_cpu_slot(block)
        self.cpu_kv_cache[:, :, cpu_block_id].copy_(
            self.kv_cache[:, :, block.gpu_block_id],
            non_blocking=False,
        )
        block.dirty = False

    def load_block_to_gpu(self, block, protected_logical_block_ids=None) -> None:
        block_manager = self._require_block_manager()
        if block.gpu_block_id is not None:
            return
        if block.cpu_block_id is None:
            raise RuntimeError(f"Logical KV block {block.logical_block_id} is not resident on CPU or GPU")
        gpu_block_id = block_manager.allocate_gpu_slot(block, protected_logical_block_ids)
        self.kv_cache[:, :, gpu_block_id].copy_(
            self.cpu_kv_cache[:, :, block.cpu_block_id],
            non_blocking=False,
        )

    def ensure_blocks_on_gpu(self, logical_block_ids: list[int], protected_logical_block_ids=None) -> None:
        block_manager = self._require_block_manager()
        for logical_block_id in dict.fromkeys(logical_block_ids):
            block = block_manager.blocks[logical_block_id]
            if block.gpu_block_id is None:
                if not block_manager.ensure_free_gpu_blocks(1, protected_logical_block_ids):
                    raise RuntimeError("No GPU KV block can be evicted for KV reload")
            self.load_block_to_gpu(block, protected_logical_block_ids)

    def build_gpu_block_table(self, seq: Sequence) -> list[int]:
        self.ensure_blocks_on_gpu(seq.block_table)
        block_manager = self._require_block_manager()
        gpu_block_table = []
        for logical_block_id in seq.block_table:
            gpu_block_id = block_manager.blocks[logical_block_id].gpu_block_id
            if gpu_block_id is None:
                raise RuntimeError(f"Logical KV block {logical_block_id} is not resident on GPU")
            gpu_block_table.append(gpu_block_id)
        return gpu_block_table
