from __future__ import annotations

import importlib.metadata
import os
import sys
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


def make_network_factory(hidden_layers, activation):
    try:
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise RuntimeError("Network factory requires Brax.") from exc
    return lambda *args, **kwargs: ppo_networks.make_ppo_networks(
        *args,
        policy_hidden_layer_sizes=hidden_layers_tuple(hidden_layers),
        activation=activation_fn(activation),
        **kwargs,
    )
