"""Server lifecycle utilities for sglang inference."""

from __future__ import annotations

from openai import OpenAI
from sglang.utils import wait_for_server, terminate_process


def launch_server(model_name: str, num_gpus: int = 1, seed: int = 42):
    import os
    os.environ['CC'] = 'gcc-10'
    os.environ['CXX'] = 'g++-10'
    os.environ['CUDAHOSTCXX'] = 'g++-10'

    # Strip DDP/torchrun env vars so the sglang subprocess doesn't try
    # to join the training process group.
    _DDP_VARS = [
        'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK',
        'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
        'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT',
        'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_USE_AGENT_STORE',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING',
    ]
    saved = {k: os.environ.pop(k) for k in _DDP_VARS if k in os.environ}

    from sglang.utils import launch_server_cmd
    import nest_asyncio
    nest_asyncio.apply()

    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {model_name} "
        f"--tool-call-parser qwen "
        f"--host 0.0.0.0 --tensor-parallel-size {num_gpus} --random-seed {seed} "
        f"--disable-cuda-graph --mem-fraction-static 0.5"
    )

    # Restore DDP env vars for the training process.
    os.environ.update(saved)

    wait_for_server(f"http://localhost:{port}")
    print(f"Server is running on http://localhost:{port}")
    return server_process, port


def setup_server_and_client(model_name, num_gpus, seed=42):
    """
    Set up the server and client for model inference.

    Returns:
        Tuple of (server_process, port, client)
    """
    server_process, port = launch_server(model_name, num_gpus=num_gpus, seed=seed)
    client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
    return server_process, port, client


def cleanup_server(server_process):
    """Clean up the server process."""
    terminate_process(server_process)
    print("Server killed")
