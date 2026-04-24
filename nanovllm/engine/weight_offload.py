from __future__ import annotations

from torch import nn


class WeightOffloadManager:

    def __init__(self, layers: nn.ModuleList, num_offload_layers: int, window_size: int = 1):
        self.layers = layers
        self.num_layers = len(layers)
        self.num_offload_layers = min(max(num_offload_layers, 0), self.num_layers)
        self.window_size = max(window_size, 1)
        self.offload_start = self.num_layers - self.num_offload_layers
        self.active_layers: set[int] = set()

    @property
    def enabled(self) -> bool:
        return self.num_offload_layers > 0

    def initialize(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            self._move_layer(layer, "cpu" if self.is_offloaded(layer_idx) else "cuda")

    def is_offloaded(self, layer_idx: int) -> bool:
        return self.enabled and layer_idx >= self.offload_start

    def begin_forward(self) -> None:
        if not self.enabled:
            return
        self.active_layers.clear()

    def prepare_layer(self, layer_idx: int) -> nn.Module:
        if not self.is_offloaded(layer_idx):
            return self.layers[layer_idx]
        target_layers = self._target_layers(layer_idx)
        stale_layers = self.active_layers - target_layers
        incoming_layers = target_layers - self.active_layers
        # Evict stale layers before pulling new ones in so the runtime peak
        # residency matches the planned window size.
        self._move_layers_to_cpu(stale_layers)
        self._move_layers_to_cuda(incoming_layers)
        self.active_layers = target_layers
        return self.layers[layer_idx]

    def end_forward(self) -> None:
        if not self.enabled or not self.active_layers:
            return
        self._move_layers_to_cpu(self.active_layers)
        self.active_layers.clear()

    def _target_layers(self, layer_idx: int) -> set[int]:
        end_idx = min(self.num_layers, layer_idx + self.window_size)
        return {idx for idx in range(layer_idx, end_idx) if self.is_offloaded(idx)}

    def _move_layers_to_cuda(self, layer_indices: set[int]) -> None:
        for layer_idx in sorted(layer_indices):
            self._move_layer(self.layers[layer_idx], "cuda")

    def _move_layers_to_cpu(self, layer_indices: set[int]) -> None:
        for layer_idx in sorted(layer_indices):
            self._move_layer(self.layers[layer_idx], "cpu")

    def _move_layer(self, layer: nn.Module, device: str) -> None:
        layer.to(device)
        self._keep_runtime_buffers_on_cuda(layer)

        #rope的layer必须一直在GPU内

    def _keep_runtime_buffers_on_cuda(self, layer: nn.Module) -> None:
        self_attn = getattr(layer, "self_attn", None)
        rotary_emb = getattr(self_attn, "rotary_emb", None)
        if rotary_emb is not None:
            rotary_emb.to("cuda")
