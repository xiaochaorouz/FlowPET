"""Model registry for the released FlowPET architecture."""

from typing import Any, Callable, Dict


class ModelRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable):
            self._registry[name] = func
            return func
        return decorator

    def create_model(self, name: str, p: Dict[str, Any], imaging_system=None, tokenizer=None):
        if name not in self._registry:
            raise ValueError(f"Unknown model: {name}. This release provides FlowPET only.")
        return self._registry[name](p, imaging_system, tokenizer)

model_registry = ModelRegistry()


@model_registry.register('FlowPET')
def create_flowpet_model(p: Dict[str, Any], imaging_system=None, tokenizer=None):
    """Build the original conditioned dual-network FlowPET model."""
    from models.unet.flowpet import FlowPET
    kwargs = p.get('backbone_kwargs', {})
    image_channels = kwargs.get('inp_channels', 1)
    condition_channels = image_channels + 2 if p.get('count_conditioning', False) else image_channels
    return FlowPET(
        image_channels=image_channels,
        model_channels=kwargs.get('model_channels', 128),
        num_res_blocks=kwargs.get('num_res_blocks', 2),
        attention_resolutions=kwargs.get('attention_resolutions', (16, 8)),
        channel_mult=kwargs.get('channel_mult', (1, 2, 4, 8)),
        use_checkpoint=kwargs.get('use_checkpoint', False),
        num_heads=kwargs.get('num_heads', 1),
        dropout=kwargs.get('dropout', 0.0),
        use_condition=kwargs.get('use_condition', True),
        adapter_channels=kwargs.get('adapter_channels'),
        condition_channels=condition_channels,
    )
