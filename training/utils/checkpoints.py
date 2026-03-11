"""Checkpointing, JSON I/O, and memory utilities."""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def free_memory(accelerator):
    """Free memory."""
    accelerator.free_memory()
    torch.cuda.empty_cache()
    accelerator.free_memory()


def save_checkpoint(model, optimizer, model_save_path, opt_save_path):
    model.save_pretrained(model_save_path, from_pt=True)
    torch.save(optimizer.state_dict(), opt_save_path)


def load_checkpoint(optimizer, model_save_path, opt_save_path):
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    optimizer.load_state_dict(torch.load(opt_save_path))
    return model, optimizer
