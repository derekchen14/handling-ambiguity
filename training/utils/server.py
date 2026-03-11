"""Server lifecycle utilities for sglang inference."""

from __future__ import annotations

from openai import OpenAI
from sglang.utils import wait_for_server, terminate_process


def launch_server(model_name: str, num_gpus: int = 1, seed: int = 42):
    from sglang.test.test_utils import is_in_ci
    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd
        import nest_asyncio
        nest_asyncio.apply()

    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {model_name} "
        f"--tool-call-parser qwen25 --reasoning-parser qwen3 "
        f"--host 0.0.0.0 --tensor-parallel-size {num_gpus} --random-seed {seed}"
    )
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
