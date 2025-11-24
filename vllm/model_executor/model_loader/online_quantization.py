# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import MethodType
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

logger = init_logger(__name__)
ONLINE_RELOAD_QUANT_METHODS = {"torchao", "awq", "awq_marlin"}

# Notes for Online Quantization
# In terms of state of checkpoints, quantization config and their
# correspondance to online quantization:
# | Use Case      | Checkpoints          |  model_config.quantization |
# | no quant      | high precision       |  None   |
# | offline quant | quantized |  fp8, torchao etc. |
# | online quant  | high precision | torchao etc. |
#
# The process for loading non-quantized checkpoint
# 1. load non-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)
#
# The process for loading offline quantized checkpoint
# 1. load offline-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)

# The process for unquantized model reloading
# (repeated run in RL training loop)
# first run
#   UI1. load_weights: load bfloat16 weights
#   UI2. process_weights_after_loading: any additional post processing
# subsequent run
#   UC1: load_weights: load bfloat16 weights
#      (shouldn't be any issues since we didn't change any attributes
#       of the weights)
#   UC2: process_weights_after_loading: any additional post processing

# The process for weight reloading with online quantization
# (repeated run in RL training loop)
# first run
#  I1. load_weights: load bfloat16 weights
#  I2. process_weights_after_loading:
#        record weight metadata and attributes for R1 and R2
#        quantize weights to fp8
# subsequent run
#  (beginning model weight is in fp8)
#  load_weights:
#    R1. restore bfloat16 model weight metadata
#    R2. restore the model weight attributes
#    R3. reload bfloat16 weights
#    R4. quantize weights (by calling process_weights_after_loading),
#    also set `process_weights_after_loading_already_called` to
#    True to stop it from running again
#    R5. (workaround for cudagraph), we restore the weight params to original quantized
#    weights params, and use original_weight_param.copy_(updated_weight_param) so that
#    the weight update work well with cudagraph
#  process_weights_after_loading (if called):
#    this will be skipped since it's already ran in
#    load_weights


def record_weights_for_reloading(
    model: nn.Module, model_config: ModelConfig
) -> None:
    if getattr(model, "weight_metadata_and_attr_saved", False):
        return

    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    quant_method = getattr(model_config, "quantization", None)
    quant_config = get_quant_config(model_config, None)
    if quant_config.get_name() not in ONLINE_RELOAD_QUANT_METHODS:
        return
    if quant_method == "torchao":
        if not (
            hasattr(quant_config, "is_checkpoint_torchao_serialized")
            and not quant_config.is_checkpoint_torchao_serialized
        ):
            return

    model.weight_loading_metadata = {
        name: _copy_to_meta_tensor(param)
        for name, param in model.named_parameters(remove_duplicate=False)
    }
    model._model_config = model_config
    model.weight_metadata_and_attr_saved = True


def restore_weights_for_loading(model: nn.Module) -> None:
    assert hasattr(model, "weight_loading_metadata")
    metadata: dict[str, torch.Tensor] = model.weight_loading_metadata
    named_modules = dict(model.named_modules(remove_duplicate=False))
    current_params = dict(model.named_parameters(remove_duplicate=False)).keys()

    for name in list(current_params):
        if name not in metadata:
            module_name, param_name = name.rsplit(".", 1)
            module = named_modules[module_name]
            delattr(module, param_name)

    for name, meta_tensor in metadata.items():
        module_name, param_name = name.rsplit(".", 1)
        module = named_modules.get(module_name)
        if module is None:
            continue

        current_param = getattr(module, param_name, None)
        if _tensors_alike(current_param, meta_tensor):
            continue

        param = _materialize_meta_tensor(meta_tensor)
        setattr(module, param_name, param)


def _copy_to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    meta_tensor = tensor.to("meta")
    meta_tensor.__class__ = tensor.__class__

    attr_state: dict[str, tuple[str, object]] = {}
    for key, value in tensor.__dict__.items():
        if isinstance(value, MethodType) and value.__self__ is tensor:
            attr_state[key] = ("method", value.__func__)
            continue
        try:
            attr_state[key] = ("value", deepcopy(value))
        except Exception:
            attr_state[key] = ("value", value)

    setattr(meta_tensor, "_original_device", tensor.device)
    setattr(meta_tensor, "_attr_state", attr_state)
    return meta_tensor


def _tensors_alike(
    tensor: torch.Tensor | None, meta_tensor: torch.Tensor
) -> bool:
    if tensor is None:
        return False

    return (
        tensor.device
        == getattr(meta_tensor, "_original_device", meta_tensor.device)
        and tensor.dtype == meta_tensor.dtype
        and tensor.shape == meta_tensor.shape
    )


def _materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    original_device = getattr(meta_tensor, "_original_device", meta_tensor.device)
    target_device = (
        torch.device("cpu")
        if isinstance(original_device, torch.device)
        and original_device.type == "cuda"
        else original_device
    )
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        device=target_device,
        requires_grad=meta_tensor.requires_grad,
    )
    tensor.__class__ = meta_tensor.__class__
    attr_state = getattr(meta_tensor, "_attr_state", {})
    for key, (kind, value) in attr_state.items():
        if kind == "method":
            setattr(tensor, key, MethodType(value, tensor))
        else:
            try:
                setattr(tensor, key, deepcopy(value))
            except Exception:
                setattr(tensor, key, value)
    return tensor


def support_quantized_model_reload_from_hp_weights(original_load_weights):
    """Decorator for `load_weights` method for AutoWeightsLoader.load_weights to support
    reloading high precision (bfloat16/float16/float32) weight for an already quantized
    model, this involves restoring the weights to a high precision weights and
    then online quantize the weights
    """
    # online quantization, right now only enabled for
    # torchao
    # R1, R2, R3, R4, R5 in the Notes

    def patched_model_load_weights(
        auto_weight_loader, weights: Iterable[tuple[str, torch.Tensor]], *, mapper=None
    ) -> set[str]:
        model = auto_weight_loader.module
        if not getattr(model, "weight_metadata_and_attr_saved", False):
            return original_load_weights(auto_weight_loader, weights, mapper=mapper)

        model_config = getattr(model, "_model_config", None)
        quant_method = getattr(model_config, "quantization", None)
        if quant_method not in ONLINE_RELOAD_QUANT_METHODS:
            return original_load_weights(auto_weight_loader, weights, mapper=mapper)

        # Step R1: First restore the quantized weights to original bfloat16
        # weights, with original metadata (shape, dtype, device)
        # and attributes, so that bfloat16 weights can be loaded properly
        original_quantized_weight_dict: dict[
            str, tuple[torch.nn.Parameter, torch.device]
        ] = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            original_device = param.device
            if original_device.type == "cuda":
                param.data = param.data.cpu()
            original_quantized_weight_dict[name] = (param, original_device)
        named_modules = dict(model.named_modules(remove_duplicate=False))

        restore_weights_for_loading(model)

        # Step R3: reload bfloat16 / high precision weights
        updated_params = original_load_weights(
            auto_weight_loader, weights, mapper=mapper
        )

        # Step R4: online quantize the weights
        # manually process weights after loading
        model.process_weights_after_loading_already_called = False
        model_device = None
        if original_quantized_weight_dict:
            first_param = next(iter(original_quantized_weight_dict.values()))
            model_device = first_param[1]

        if model_device is not None:
            process_weights_after_loading(model, model_config, model_device)
        else:
            logger.warning_once(
                "model_device is None, skip calling process_weights_after_loading"
            )

        # Step R5 (workaround for cudagraph): restore the original quantized weights
        # and do a copy_ of the currents weights to the original weights
        updated_quantized_weights = dict(model.named_parameters(remove_duplicate=False))
        for name, (original_quantized_weight, original_device) in (
            original_quantized_weight_dict.items()
        ):
            updated_quantized_weight = updated_quantized_weights.get(name)
            if updated_quantized_weight is None:
                continue

            module_name, weight_name = name.rsplit(".", 1)
            module = named_modules[module_name]
            setattr(module, weight_name, original_quantized_weight)
            if original_quantized_weight.device != original_device:
                original_quantized_weight.data = original_quantized_weight.data.to(
                    original_device
                )
            with torch.no_grad():
                original_quantized_weight.copy_(updated_quantized_weight)

        del original_quantized_weight_dict
        del named_modules

        model.process_weights_after_loading_already_called = True
        return updated_params

    return patched_model_load_weights
