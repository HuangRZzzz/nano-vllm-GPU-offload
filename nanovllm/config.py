import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_mixed_batch: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    min_num_kvcache_blocks: int | None = None
    max_num_kvcache_blocks: int | None = None
    min_kvcache_memory_bytes: int = 1024**3
    cpu_offload_num_layers: int | str = 0
    cpu_offload_window_size: int = 1
    enable_kv_offload: bool = False
    cpu_kvcache_gb: float = 0.0
    num_cpu_kvcache_blocks: int | None = None
    kv_offload_policy: str = "lru"
    kv_offload_watermark: float = 0.9
    kv_prefetch_sync: bool = True
    kv_offload_async: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens > 0
        if isinstance(self.cpu_offload_num_layers, str):
            value = self.cpu_offload_num_layers.strip().lower()
            self.cpu_offload_num_layers = value if value == "auto" else int(value)
        assert self.cpu_offload_num_layers == "auto" or self.cpu_offload_num_layers >= 0
        assert self.cpu_offload_window_size > 0
        assert self.min_num_kvcache_blocks is None or self.min_num_kvcache_blocks > 0
        assert self.max_num_kvcache_blocks is None or self.max_num_kvcache_blocks > 0
        assert self.min_kvcache_memory_bytes >= 0
        assert self.cpu_kvcache_gb >= 0
        assert self.num_cpu_kvcache_blocks is None or self.num_cpu_kvcache_blocks > 0
        assert 0 < self.kv_offload_watermark <= 1
        if self.min_num_kvcache_blocks is not None and self.max_num_kvcache_blocks is not None:
            assert self.min_num_kvcache_blocks <= self.max_num_kvcache_blocks
        if self.enable_kv_offload:
            assert self.tensor_parallel_size == 1, "KV offload MVP currently supports tensor_parallel_size=1 only"
            assert self.kv_offload_policy == "lru"
            assert self.kv_prefetch_sync
            assert not self.kv_offload_async
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
