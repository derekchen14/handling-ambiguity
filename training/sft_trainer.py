"""SFT training loop for NLU pipeline stages.

Provides ``NLUSFTDataset`` (torch Dataset) and ``run_sft`` which runs a
standard cross-entropy training loop with label masking on assistant tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.sft_data import SFTDataGenerator, SFTExample
from training.stages import PipelineStage


class NLUSFTDataset(Dataset):
    """Tokenised SFT dataset with label masking for assistant tokens.

    Mirrors the tokenisation pattern from ``RLTrainingDataset`` but produces
    cross-entropy labels instead of advantages.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        tool_specs: list[dict] | None = None,
    ):
        """
        Args:
            examples: Chat-format dicts with ``messages`` key (from SFTExample.to_chat_format).
            tokenizer: HuggingFace tokenizer.
            max_length: Max sequence length (pad/truncate).
            tool_specs: Optional tool specifications for ``apply_chat_template``.
        """
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
            token_ids = self.tokenizer.apply_chat_template(
                messages, tools=self.tool_specs, tokenize=True,
            )
        except Exception:
            return None

        length = len(token_ids)
        attention_mask = torch.ones(length, dtype=torch.int64)

        # Pad or truncate
        if length < self.max_length:
            pad_len = self.max_length - length
            token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.int64),
            ])
        else:
            token_ids = token_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            length = self.max_length

        input_ids = torch.tensor(token_ids, dtype=torch.int64)

        # Build label mask: only train on assistant tokens
        # Use the Qwen convention: <|im_start|>=151644, assistant=77091, <|im_end|>=151645
        label_mask = self._build_assistant_mask(input_ids)

        # Labels = input_ids shifted by 1 (standard causal LM), masked elsewhere
        labels = input_ids.clone()
        labels[label_mask == 0] = -100  # Ignore non-assistant tokens

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'label_mask': label_mask,
        }

    @staticmethod
    def _build_assistant_mask(input_ids: torch.Tensor) -> torch.Tensor:
        """Identify assistant token spans.

        Looks for the pattern ``<|im_start|>assistant`` (token IDs 151644, 77091)
        and marks all tokens until the next ``<|im_end|>`` (151645).
        Falls back to a heuristic if the specific token IDs aren't found.
        """
        mask = torch.zeros_like(input_ids)
        ids = input_ids.tolist()
        i = 0
        while i < len(ids) - 1:
            if ids[i] == 151644 and ids[i + 1] == 77091:
                # Mark from after "assistant\n" to <|im_end|>
                start = i + 2
                end = len(ids)
                for j in range(start, len(ids)):
                    if ids[j] == 151645:
                        end = j + 1
                        break
                mask[start:end] = 1
                i = end
            else:
                i += 1
        return mask

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encodings[idx]


def _collate_sft(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for SFT DataLoader."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'label_mask': torch.stack([b['label_mask'] for b in batch]),
    }


def run_sft(
    args,
    stage: PipelineStage,
    eval_set: list[dict],
    tools: list[dict] | None = None,
) -> None:
    """Run supervised fine-tuning for a pipeline stage.

    Args:
        args: Parsed argument namespace (from train_nlu.py) with fields:
            model_name, sft_epochs, sft_lr, max_tokens, batch_size,
            model_save_path, wandb_project, wandb_name, confidence_threshold,
            ensemble_results_path, domain.
        stage: The pipeline stage to train.
        eval_set: The evaluation set (list of conversation dicts).
        tools: Optional tool manifest (for tool-calling stages).
    """
    accelerator = Accelerator(mixed_precision='fp16')

    # Build stage kwargs
    stage_kwargs: dict[str, Any] = {}
    if tools:
        stage_kwargs['tools'] = tools

    # Generate SFT examples
    generator = SFTDataGenerator(stage, args.domain, **stage_kwargs)

    if args.ensemble_results_path:
        ensemble_results = SFTDataGenerator.load_jsonl(args.ensemble_results_path)
        examples = generator.generate_from_ensemble_results(
            ensemble_results, eval_set, args.confidence_threshold
        )
    else:
        examples = generator.generate_from_gold_labels(eval_set)

    if not examples:
        print('No SFT examples generated. Check eval set and stage configuration.')
        return

    chat_examples = [ex.to_chat_format() for ex in examples]
    print(f'Generated {len(chat_examples)} SFT examples for stage={stage.name}')

    # Initialise tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tool_specs = None
    if tools:
        from prompts.tool_calling import strip_tool_metadata
        tool_specs = strip_tool_metadata(tools)

    dataset = NLUSFTDataset(chat_examples, tokenizer, args.max_tokens, tool_specs)
    print(f'Tokenised {len(dataset)} examples (max_length={args.max_tokens})')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_sft,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Wandb logging
    if accelerator.is_main_process:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    'mode': 'sft',
                    'stage': stage.name,
                    'domain': args.domain,
                    'model': args.model_name,
                    'sft_lr': args.sft_lr,
                    'sft_epochs': args.sft_epochs,
                    'num_examples': len(dataset),
                },
            )
        except ImportError:
            wandb = None
    else:
        wandb = None

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(args.sft_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            loss = outputs.loss
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.detach().float().item()
            num_batches += 1
            global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            print(f'Epoch {epoch + 1}/{args.sft_epochs} — loss: {avg_loss:.4f}')
            if wandb is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'sft_loss': avg_loss,
                    'global_step': global_step,
                })

    # Save
    if accelerator.is_main_process:
        save_dir = Path(args.model_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f'Model saved to {save_dir}')

        if wandb is not None:
            wandb.finish()
