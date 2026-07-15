from __future__ import annotations

import importlib.metadata
import os
import sys
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from pathlib import Path


def hidden_layers_tuple(values):
    return tuple(int(value) for value in values)


def select_mujoco_gl_backend(environ=None, platform_name=None):
    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    configured = environ.get("MUJOCO_GL")
    if configured:
        return configured
    if platform_name.startswith("linux") and not environ.get("DISPLAY"):
        return "egl"
    return "glfw"


def configure_cloud_runtime(xla_triton=True, mujoco_gl="auto", matmul_precision="high", verbose=False):
    flags = os.environ.get("XLA_FLAGS", "")
    xla_flags = []
    if xla_triton:
        xla_flags.extend(
            [
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                "--xla_gpu_shard_autotuning=false",
                "--xla_gpu_triton_gemm_any=True",
            ]
        )
        try:
            jaxlib_version = importlib.metadata.version("jaxlib").replace(".", "_")
        except importlib.metadata.PackageNotFoundError:
            jaxlib_version = None
        if jaxlib_version is not None:
            autotune_path = f"/tmp/xla_autotune_jaxlib_{jaxlib_version}.pbtxt"
            xla_flags.append(f"--xla_gpu_dump_autotune_results_to={autotune_path}")
            if Path(autotune_path).exists():
                xla_flags.append(f"--xla_gpu_load_autotune_results_from={autotune_path}")
    for flag in xla_flags:
        if flag not in flags:
            flags = f"{flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = flags

    if mujoco_gl == "auto":
        os.environ["MUJOCO_GL"] = select_mujoco_gl_backend()
    elif mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
    if os.environ.get("MUJOCO_GL"):
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]

    if matmul_precision:
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", matmul_precision)
    if verbose:
        print(
            "stage=runtime_config "
            f"mujoco_gl={os.environ.get('MUJOCO_GL', '')} "
            f"matmul_precision={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '')} "
            f"xla_flags={os.environ.get('XLA_FLAGS', '')}",
            flush=True,
        )


def activation_fn(name):
    try:
        import jax.nn as jnn
    except ImportError as exc:
        raise RuntimeError("Activation functions require JAX.") from exc
    return {
        "relu": jnn.relu,
        "tanh": jnn.tanh,
        "elu": jnn.elu,
        "swish": jnn.swish,
        "silu": jnn.silu,
    }[name]


def _replace_field(value, **changes):
    if is_dataclass(value):
        return replace(value, **changes)
    if hasattr(value, "_replace"):
        return value._replace(**changes)
    return type(value)(**{**vars(value), **changes})


def _replace_mapping_value(mapping, key, value):
    copied = dict(mapping)
    copied[key] = value
    try:
        return type(mapping)(copied)
    except TypeError:
        return mapping.copy(add_or_replace={key: value})


def _zero_policy_output_layer(params, zeros_like):
    """Zeros the final layer of a Brax MLP parameter tree across 0.10+ layouts."""

    if isinstance(params, Mapping):
        if "params" in params:
            return _replace_mapping_value(
                params,
                "params",
                _zero_policy_output_layer(params["params"], zeros_like),
            )
        layer_keys = [
            key
            for key, value in params.items()
            if isinstance(value, Mapping) and ("kernel" in value or "bias" in value)
        ]
        if layer_keys:
            key = layer_keys[-1]
            return _replace_mapping_value(
                params, key, _tree_zeros(params[key], zeros_like)
            )
    if isinstance(params, (list, tuple)) and params:
        values = list(params)
        values[-1] = _tree_zeros(values[-1], zeros_like)
        return type(params)(values) if isinstance(params, tuple) else values
    raise RuntimeError(
        "Unsupported Brax policy parameter layout; cannot zero the actor output layer"
    )


def _tree_zeros(value, zeros_like):
    if isinstance(value, Mapping):
        return type(value)(
            {key: _tree_zeros(item, zeros_like) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_tree_zeros(item, zeros_like) for item in value]
    if isinstance(value, tuple):
        return type(value)(_tree_zeros(item, zeros_like) for item in value)
    return zeros_like(value)


def make_network_factory(hidden_layers, activation, zero_policy_output=False):
    try:
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise RuntimeError("Network factory requires Brax.") from exc

    def factory(*args, **kwargs):
        networks = ppo_networks.make_ppo_networks(
            *args,
            policy_hidden_layer_sizes=hidden_layers_tuple(hidden_layers),
            activation=activation_fn(activation),
            **kwargs,
        )
        if not zero_policy_output:
            return networks

        import jax

        policy_network = networks.policy_network
        original_init = policy_network.init

        def zero_output_init(key):
            return _zero_policy_output_layer(original_init(key), jax.numpy.zeros_like)

        policy_network = _replace_field(policy_network, init=zero_output_init)
        return _replace_field(networks, policy_network=policy_network)

    return factory
