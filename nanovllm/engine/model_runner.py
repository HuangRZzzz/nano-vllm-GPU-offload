import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.weight_offload import WeightOffloadManager
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.cpu_offload_auto = config.cpu_offload_num_layers == "auto"
        self.cpu_offload_requested = self.cpu_offload_auto or config.cpu_offload_num_layers > 0
        self.cpu_offload_enabled = False
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cpu" if self.cpu_offload_requested else "cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.reserve_min_kv_cache()
        self.init_weight_offload()
        torch.set_default_device("cpu" if self.cpu_offload_enabled else "cuda")
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if hasattr(self, "kv_cache"):
            del self.kv_cache
        if hasattr(self, "kv_cache_reserve"):
            del self.kv_cache_reserve
        if hasattr(self, "sampler"):
            del self.sampler
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        warmup_len = min(self.config.max_model_len, self.config.max_num_batched_tokens)
        num_seqs = min(max(1, self.config.max_num_batched_tokens // warmup_len), self.config.max_num_seqs)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.scheduled_prefill_end = len(seq)
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def                                                                                                                                                                                                          init_weight_offload(self):
        model = self.model.model
        model.embed_tokens.to("cuda")
        model.norm.to("cuda")
        self.model.lm_head.to("cuda")
        if self.config.hf_config.tie_word_embeddings:
            self.model.lm_head.weight.data = model.embed_tokens.weight.data
        num_offload_layers = self.resolve_num_offload_layers(model)
        self.weight_offload_manager = WeightOffloadManager(
            model.layers,
            num_offload_layers,
            self.config.cpu_offload_window_size,
        )
        if self.cpu_offload_requested:
            self.weight_offload_manager.initialize()
        self.cpu_offload_enabled = self.weight_offload_manager.enabled
        self.enforce_eager = self.config.enforce_eager or self.cpu_offload_enabled
        if self.cpu_offload_enabled:
            model.weight_offload_manager = self.weight_offload_manager
        torch.cuda.empty_cache()

    def resolve_num_offload_layers(self, model) -> int:
        if not self.cpu_offload_auto:
            return self.config.cpu_offload_num_layers
        num_offload_layers = self.plan_auto_num_offload_layers(model)
        if self.rank == 0:
            num_layers = len(model.layers)
            num_gpu_layers = num_layers - num_offload_layers
            print(
                f"[nano-vllm] auto CPU offload: keeping {num_gpu_layers}/{num_layers} "
                f"decoder layers on CUDA, offloading {num_offload_layers} to CPU"
            )
        return num_offload_layers

    def plan_auto_num_offload_layers(self, model) -> int:
        layer_sizes = [self.module_parameter_bytes(layer) for layer in model.layers]
        num_layers = len(layer_sizes)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        budget = max(0, int(total * self.config.gpu_memory_utilization) - used)
        kv_reserve = self.estimate_kv_cache_reserve_bytes()
        runtime_buffer_reserve = self.module_buffer_bytes(model.layers)
        workspace_reserve = max(256 * 1024**2, int(total * 0.05))
        layer_budget = max(0, budget - kv_reserve - runtime_buffer_reserve - workspace_reserve)

        prefix_bytes = [0]
        for size in layer_sizes:
            prefix_bytes.append(prefix_bytes[-1] + size)

        window_size = max(1, self.config.cpu_offload_window_size)
        for num_gpu_layers in range(num_layers, -1, -1):
            static_bytes = prefix_bytes[num_gpu_layers]
            active_window_bytes = 0
            for layer_idx in range(num_gpu_layers, num_layers):
                end_idx = min(num_layers, layer_idx + window_size)
                active_window_bytes = max(
                    active_window_bytes,
                    prefix_bytes[end_idx] - prefix_bytes[layer_idx],
                )
            if static_bytes + active_window_bytes <= layer_budget:
                return num_layers - num_gpu_layers
        return num_layers

    def estimate_kv_cache_reserve_bytes(self) -> int:
        if getattr(self, "kv_cache_reserve", None) is not None:
            return 0
        return self.min_kvcache_blocks() * self.kv_cache_block_bytes()

    def reserve_min_kv_cache(self) -> None:
        reserve_bytes = self.min_kvcache_blocks() * self.kv_cache_block_bytes()
        self.kv_cache_reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device="cuda")

    def release_kv_cache_reserve(self) -> None:
        if getattr(self, "kv_cache_reserve", None) is None:
            return
        del self.kv_cache_reserve
        torch.cuda.empty_cache()

    def min_kvcache_blocks(self) -> int:
        block_bytes = self.kv_cache_block_bytes()
        num_blocks = max(1, (self.config.min_kvcache_memory_bytes + block_bytes - 1) // block_bytes)
        if self.config.min_num_kvcache_blocks is not None:
            num_blocks = max(num_blocks, self.config.min_num_kvcache_blocks)
        if self.config.max_num_kvcache_blocks is not None and num_blocks > self.config.max_num_kvcache_blocks:
            raise RuntimeError(
                "Invalid KV cache limits: "
                f"minimum required blocks={num_blocks}, "
                f"max_num_kvcache_blocks={self.config.max_num_kvcache_blocks}. "
                "Increase max_num_kvcache_blocks or lower min_kvcache_memory_bytes/min_num_kvcache_blocks."
            )
        return num_blocks

    def kv_cache_block_bytes(self) -> int:
        hf_config = self.config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

    @staticmethod
    def module_parameter_bytes(module) -> int:
        return ModelRunner.unique_tensor_bytes(module.parameters())

    @staticmethod
    def module_buffer_bytes(module) -> int:
        return ModelRunner.unique_tensor_bytes(module.buffers())

    @staticmethod
    def unique_tensor_bytes(tensors) -> int:
        seen = set()
        total = 0
        for tensor in tensors:
            if tensor is None or tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
        return total

    @staticmethod
    def format_bytes(num_bytes: int) -> str:
        value = float(num_bytes)
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.2f}{unit}"
            value /= 1024

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = self.kv_cache_block_bytes()
        min_blocks = self.min_kvcache_blocks()
        reserve_active = getattr(self, "kv_cache_reserve", None) is not None
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        util_cap_bytes = int(total * config.gpu_memory_utilization)
        transient_bytes = max(0, peak - current)
        extra_bytes = 0
        if reserve_active:
            # Keep the pre-reserved KV floor, and only grow beyond it if there is
            # additional headroom after accounting for transient warmup buffers.
            extra_bytes = max(0, util_cap_bytes - used - transient_bytes)
            num_blocks = min_blocks + extra_bytes // block_bytes
        else:
            available_bytes = util_cap_bytes - used - peak + current
            num_blocks = available_bytes // block_bytes
        if config.max_num_kvcache_blocks is not None:
            num_blocks = min(num_blocks, config.max_num_kvcache_blocks)
        if num_blocks < min_blocks:
            raise RuntimeError(
                "Not enough GPU memory to allocate KV cache: "
                f"available_blocks={max(num_blocks, 0)}, required_blocks={min_blocks}, "
                f"block_bytes={block_bytes}, "
                f"min_kvcache_memory_bytes={config.min_kvcache_memory_bytes}. "
                "Try increasing cpu_offload_num_layers, "
                "using cpu_offload_num_layers='auto', lowering cpu_offload_window_size, "
                "lowering max_num_batched_tokens/max_model_len/max_num_seqs, or setting "
                "a higher max_num_kvcache_blocks."
            )
        if self.rank == 0:
            min_bytes = min_blocks * block_bytes
            final_bytes = num_blocks * block_bytes
            extra_blocks = max(0, num_blocks - min_blocks)
            print(
                "[nano-vllm] KV cache allocation:\n"
                f"  reserve_active={reserve_active}\n"
                f"  block_bytes={block_bytes} ({self.format_bytes(block_bytes)})\n"
                f"  min_blocks={min_blocks} ({self.format_bytes(min_bytes)})\n"
                f"  extra_blocks={extra_blocks} ({self.format_bytes(extra_blocks * block_bytes)})\n"
                f"  final_blocks={num_blocks} ({self.format_bytes(final_bytes)})\n"
                f"  used={self.format_bytes(used)}, peak={self.format_bytes(peak)}, "
                f"current={self.format_bytes(current)}, transient={self.format_bytes(transient_bytes)}\n"
                f"  util_cap={self.format_bytes(util_cap_bytes)}, total={self.format_bytes(total)}"
            )
        self.release_kv_cache_reserve()
        config.num_kvcache_blocks = num_blocks
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device="cuda",
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        use_block_tables = False
        for seq in seqs:
            prefill_end = seq.scheduled_prefill_end or len(seq)
            input_ids.extend(seq[seq.num_computed_tokens:prefill_end])
            positions.extend(list(range(seq.num_computed_tokens, prefill_end)))
            seqlen_q = prefill_end - seq.num_computed_tokens
            seqlen_k = prefill_end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            use_block_tables = True
            for pos in range(seq.num_computed_tokens, prefill_end):
                block_id = seq.block_table[pos // self.block_size]
                slot_mapping.append(block_id * self.block_size + pos % self.block_size)
        if use_block_tables:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_mixed(self, decode_seqs: list[Sequence], prefill_seqs: list[Sequence]):
        seqs = decode_seqs + prefill_seqs
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in decode_seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            max_seqlen_q = max(1, max_seqlen_q)
            max_seqlen_k = max(len(seq), max_seqlen_k)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        for seq in prefill_seqs:
            prefill_end = seq.scheduled_prefill_end
            input_ids.extend(seq[seq.num_computed_tokens:prefill_end])
            positions.extend(list(range(seq.num_computed_tokens, prefill_end)))
            seqlen_q = prefill_end - seq.num_computed_tokens
            seqlen_k = prefill_end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for pos in range(seq.num_computed_tokens, prefill_end):
                block_id = seq.block_table[pos // self.block_size]
                slot_mapping.append(block_id * self.block_size + pos % self.block_size)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return seqs, input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def run_mixed(self, decode_seqs: list[Sequence], prefill_seqs: list[Sequence]) -> list[int]:
        seqs, input_ids, positions = self.prepare_mixed(decode_seqs, prefill_seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, True)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
