# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import MethodType

import torch

_ATTRS_TO_SKIP = {
    "grad",
    "_grad",
    "_grad_fn",
    "_is_replica",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_pre_hooks",
    "_non_persistent_buffers_set",
}


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(
    mod: torch.nn.Module, name: str, new: torch.Tensor | torch.nn.Parameter
) -> None:
    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter when metadata changes
        new_param = new if isinstance(new, torch.nn.Parameter) else None
        if new_param is None:
            new_param = torch.nn.Parameter(new, requires_grad=old.requires_grad)
        _copy_parameter_metadata(old, new_param)
        mod.register_parameter(name, new_param)


def _copy_parameter_metadata(old: torch.nn.Parameter,
                             new: torch.nn.Parameter) -> None:
    """Copy custom attributes from the original parameter to the replacement."""
    for attr, value in old.__dict__.items():
        if attr in _ATTRS_TO_SKIP:
            continue
        if isinstance(value, MethodType) and value.__self__ is old:
            value = MethodType(value.__func__, new)
        setattr(new, attr, value)
