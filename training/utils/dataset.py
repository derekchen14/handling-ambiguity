"""Dataset and sampling utilities for RL and SFT training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def tokenize_and_pad(tokenizer, messages, max_length, tool_specs=None):
    """Shared tokenization: apply_chat_template -> pad/truncate -> (input_ids, attention_mask, length)."""
    token_ids = tokenizer.apply_chat_template(messages, tools=tool_specs, tokenize=True)
    length = len(token_ids)
    attention_mask = torch.ones(length, dtype=torch.int64)

    if length < max_length:
        pad_len = max_length - length
        token_ids = token_ids + [tokenizer.pad_token_id or 0] * pad_len
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.int64)])
    else:
        token_ids = token_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        length = max_length

    input_ids = torch.tensor(token_ids, dtype=torch.int64)
    return input_ids, attention_mask, length


def find_assistant_spans(input_ids, im_start=151644, assistant=77091, im_end=151645):
    """Find (start, end) index pairs for assistant token spans."""
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    ids = input_ids.tolist()
    spans = []
    i = 0
    while i < len(ids) - 1:
        if ids[i] == im_start and ids[i + 1] == assistant:
            start = i + 2
            end = len(ids)
            for j in range(start, len(ids)):
                if ids[j] == im_end:
                    end = j + 1
                    break
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# RL dataset
# ---------------------------------------------------------------------------

class RLTrainingDataset(Dataset):
    """A PyTorch dataset wrapper for HuggingFace datasets."""

    def __init__(
        self,
        trajectories: List[List[Dict[str, Any]]],
        tool_specs: List[Dict[str, Any]],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 131072,
        actor_critic: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.all_encodings = []
        self.encodings = []
        self.advantages = []
        self.group_ids = []
        self.reference_probas_dir = None
        self.old_policy_probas_dir = None
        self.trajectories = trajectories
        self.tool_specs = tool_specs
        self.actor_critic = actor_critic
        self.rewards = []

        group_idx = 0
        for point in trajectories:
            if not point:
                continue

            reward_list = []
            message_list = []
            for traj in point:
                reward_list.append(traj['reward'])
                message_list.append(traj['messages'])

            reward_arr = np.array(reward_list).astype(np.float32)
            reward_arr = (reward_arr * 2) - 1

            if self.actor_critic:
                self.rewards.append(reward_arr)

            if reward_arr.std() == 0:
                reward_arr = None
            else:
                reward_arr = (reward_arr - reward_arr.mean()) / reward_arr.std()

            tokenized_messages = []
            for message in message_list:
                input_ids, attention_mask, length = tokenize_and_pad(
                    self.tokenizer, message, self.max_length, self.tool_specs
                )
                tokenized_messages.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'length': torch.tensor(length, dtype=torch.int32),
                })

            if self.actor_critic:
                self.group_ids += [group_idx] * len(message_list)
                self.encodings += tokenized_messages
                group_idx += 1
                continue

            if reward_arr is not None:
                self.advantages += reward_arr.tolist()
                self.group_ids += [group_idx] * len(message_list)
                self.encodings += tokenized_messages
                group_idx += 1
            else:
                self.all_encodings += tokenized_messages

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encodings = self.encodings[idx]

        if self.tokenizer is None:
            return encodings

        assistant_tok_indices = find_assistant_spans(encodings['input_ids'])

        label_mask = torch.zeros_like(encodings['input_ids'])
        for tok_idx in assistant_tok_indices:
            start_idx, end_idx = tok_idx
            label_mask[start_idx:end_idx] = 1

        encodings['label_mask'] = label_mask

        if not self.actor_critic:
            advantage = self.advantages[idx]
            encodings['advantage'] = torch.tensor(advantage, dtype=torch.float32)
        else:
            reward_tensor = torch.zeros(self.max_length, dtype=torch.float32)
            reward_tensor[encodings['length'] - 1] = self.rewards[idx]
            encodings['rewards'] = reward_tensor
        encodings['idx'] = torch.tensor(idx, dtype=torch.int32)
        encodings['group_id'] = torch.tensor(self.group_ids[idx], dtype=torch.int32)

        if self.reference_probas_dir:
            ref_prob = torch.load(
                f'{self.reference_probas_dir}/{idx}.pt', map_location="cpu"
            )
            encodings['ref_prob'] = ref_prob
        if self.old_policy_probas_dir:
            old_prob = torch.load(
                f'{self.old_policy_probas_dir}/{idx}.pt', map_location="cpu"
            )
            encodings['old_prob'] = old_prob
        return encodings

    def enable_reference_probas(self, reference_probas_dir: str):
        self.reference_probas_dir = reference_probas_dir

    def enable_old_policy_probas(self, old_policy_probas_dir: str):
        self.old_policy_probas_dir = old_policy_probas_dir


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

class NLUSFTDataset(Dataset):
    """Tokenised SFT dataset with label masking for assistant tokens.

    Mirrors the tokenisation pattern from ``RLTrainingDataset`` but produces
    cross-entropy labels instead of advantages.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_length: int = 2048,
        tool_specs: list[dict] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tool_specs = tool_specs
        self.encodings: list[dict[str, torch.Tensor]] = []

        for ex in examples:
            messages = ex['messages']
            enc = self._tokenize(messages)
            if enc is not None:
                self.encodings.append(enc)

    def _tokenize(self, messages: list[dict]) -> dict[str, torch.Tensor] | None:
        """Tokenize messages and create label mask for assistant tokens."""
        try:
            input_ids, attention_mask, length = tokenize_and_pad(
                self.tokenizer, messages, self.max_length, self.tool_specs
            )
        except Exception:
            return None

        # Build label mask using shared helper
        spans = find_assistant_spans(input_ids)
        label_mask = torch.zeros_like(input_ids)
        for start, end in spans:
            label_mask[start:end] = 1

        # Labels = input_ids masked elsewhere
        labels = input_ids.clone()
        labels[label_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'label_mask': label_mask,
        }

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encodings[idx]


def collate_sft(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for SFT DataLoader."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'label_mask': torch.stack([b['label_mask'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# RL sampling utilities
# ---------------------------------------------------------------------------

class GroupBatchSampler(Sampler):
    """Sampler that ensures all trajectories from the same group are in the same batch."""

    def __init__(self, group_ids, batch_size, shuffle=True):
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.groups = defaultdict(list)
        for idx, group_id in enumerate(group_ids):
            self.groups[group_id].append(idx)

        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.group_keys)

        batch = []
        for group_key in self.group_keys:
            group_indices = self.groups[group_key].copy()
            if self.shuffle:
                random.shuffle(group_indices)

            if batch and len(batch) + len(group_indices) > self.batch_size:
                yield batch
                batch = []

            batch.extend(group_indices)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def __len__(self):
        return len(self.group_keys)


def collate_group_batch(batch):
    """Custom collate function that handles variable-length trajectories."""
    trajectories = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    label_masks = torch.stack([item['label_mask'] for item in batch])
    rewards = torch.stack([item['advantage'] for item in batch])
    group_ids = torch.stack([item['group_id'] for item in batch])

    result = {
        'input_ids': trajectories,
        'advantage': rewards,
        'attention_mask': attention_masks,
        'label_mask': label_masks,
        'group_id': group_ids,
    }

    if 'ref_prob' in batch[0]:
        ref_probs = torch.stack([item['ref_prob'] for item in batch])
        result['ref_prob'] = ref_probs
    if 'old_prob' in batch[0]:
        old_probs = torch.stack([item['old_prob'] for item in batch])
        result['old_prob'] = old_probs

    return result
