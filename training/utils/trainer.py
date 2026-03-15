"""ValueHead, PPOTrainer (RL) and run_sft (SFT) training utilities."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from training.utils.losses import ppo_loss, compute_td_returns
from training.utils.checkpoints import save_checkpoint, load_checkpoint, free_memory
from training.utils.dataset import GroupBatchSampler, collate_group_batch, NLUSFTDataset, collate_sft


class ValueHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class PPOTrainer:
    def __init__(self, model_name, reference_model_name, use_value_head,
                 value_head_path, accelerator, train_dataset,
                 optimizer_path=None, batch_size=8, lr=2e-6, eps_high=0.2,
                 eps_low=0.2, beta=0.015, k_value=7, entropy_coef=None,
                 use_kl_div=False, use_tok_pg=False,
                 gradient_accumulation_steps=4, group_size=None,
                 max_grad_norm=0.9):
        self.model_name = model_name
        self.reference_model_name = reference_model_name
        self.accelerator = accelerator
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.eps_high = eps_high
        self.eps_low = eps_low
        self.beta = beta
        self.k_value = k_value
        self.optimizer_path = optimizer_path
        self.entropy_coef = entropy_coef
        self.use_kl_div = use_kl_div
        self.use_tok_pg = use_tok_pg
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.group_size = group_size
        self.max_grad_norm = max_grad_norm
        self.use_value_head = use_value_head
        self.value_head_path = value_head_path

    def compute_reference_probs(self):
        save_dir = 'reference_probas'
        self._compute_probs(self.reference_model_name, save_dir)
        self.train_dataset.enable_reference_probas(save_dir)

    def compute_old_policy_probs(self):
        save_dir = 'old_policy_probas'
        self._compute_probs(self.model_name, save_dir)
        self.train_dataset.enable_old_policy_probas(save_dir)

    def _compute_probs(self, model_name, save_dir):
        model = AutoModelForCausalLM.from_pretrained(model_name)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        model, train_dataloader = self.accelerator.prepare(model, train_dataloader)
        model.eval()

        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(train_dataloader, total=len(train_dataloader),
                              desc="Computing probabilities"):
                out = model(**batch)
                probs = out.logits.softmax(dim=2).detach()
                sampled_action_probs = torch.gather(
                    probs[:, :-1], 2,
                    batch['input_ids'][:, 1:].unsqueeze(-1)
                ).squeeze(-1)

                for idx, prob in zip(batch['idx'], sampled_action_probs):
                    torch.save(prob, f'{save_dir}/{idx}.pt')

        del model, train_dataloader, batch, out, probs, sampled_action_probs
        free_memory(self.accelerator)

    def train_epoch(self):
        sampler = GroupBatchSampler(
            group_ids=self.train_dataset.group_ids,
            batch_size=self.group_size,
            shuffle=True
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=collate_group_batch,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        try:
            import wandb
            wandb.watch(self.model, log="all")
        except ImportError:
            pass

        if self.optimizer_path:
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))

        self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )

        if self.use_value_head:
            self.value_head = ValueHead(self.model.config.hidden_size)
            if self.value_head_path:
                self.value_head.load_state_dict(torch.load(self.value_head_path))
            self.value_head = self.accelerator.prepare(self.value_head)

        self.model.train()
        total_loss = torch.tensor(0.0, device=self.model.device)
        total_kl_div = torch.tensor(0.0, device=self.model.device)
        total_entropy = torch.tensor(0.0, device=self.model.device)

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            num_subbatches = batch['input_ids'].shape[0] // self.batch_size
            if batch['input_ids'].shape[0] % self.batch_size != 0:
                num_subbatches += 1

            for subbatch_idx in range(num_subbatches):
                start_idx = subbatch_idx * self.batch_size
                end_idx = min((subbatch_idx + 1) * self.batch_size,
                              batch['input_ids'].shape[0])

                subbatch = {
                    'input_ids': batch['input_ids'][start_idx:end_idx],
                    'attention_mask': batch['attention_mask'][start_idx:end_idx],
                    'label_mask': batch['label_mask'][start_idx:end_idx],
                    'advantage': batch['advantage'][start_idx:end_idx],
                    'group_id': batch['group_id'][start_idx:end_idx]
                }

                if 'old_prob' in batch:
                    subbatch['old_prob'] = batch['old_prob'][start_idx:end_idx]
                if 'ref_prob' in batch:
                    subbatch['ref_prob'] = batch['ref_prob'][start_idx:end_idx]

                out = self.model(**subbatch, output_hidden_states=True)

                last_hidden_state = out.hidden_states[-1]
                value = self.value_head(last_hidden_state)
                value *= subbatch['attention_mask']
                advantages = compute_td_returns(value, subbatch['rewards'])

                loss, kl_div, entropy = ppo_loss(
                    out.logits[:, :-1],
                    subbatch['input_ids'][:, 1:],
                    subbatch['advantage'],
                    eps_high=self.eps_high,
                    eps_low=self.eps_low,
                    label_mask=subbatch['label_mask'][:, 1:],
                    old_policy_probs=subbatch['old_prob'] if 'old_prob' in subbatch else None,
                    reference_probs=subbatch['ref_prob'] if 'ref_prob' in subbatch else None,
                    use_kl_div=self.use_kl_div,
                    beta=self.beta,
                    entropy_coef=self.entropy_coef,
                    use_tok_pg=self.use_tok_pg
                )

                if torch.isnan(loss):
                    print('Loss is nan, stopping training')
                    continue

                prop = subbatch['input_ids'].shape[0] / len(self.train_dataset)
                loss *= prop

                self.accelerator.backward(loss)

                total_loss += loss.detach()

                if kl_div is not None:
                    kl_div *= prop
                    total_kl_div += kl_div.detach()

                if entropy is not None:
                    entropy *= prop
                    total_entropy += entropy.detach()

            if batch_idx % self.gradient_accumulation_steps == 0 or \
               batch_idx == len(train_dataloader) - 1:
                self.accelerator.clip_grad_norm_(self.model.parameters(),
                                                 self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        total_loss = self.accelerator.reduce(total_loss, reduction='sum')
        total_kl_div = self.accelerator.reduce(total_kl_div, reduction='sum')
        total_entropy = self.accelerator.reduce(total_entropy, reduction='sum')

        return total_loss, total_kl_div, total_entropy

    def save(self, model_save_path, opt_save_path, value_head_path):
        save_checkpoint(self.model, self.optimizer, model_save_path, opt_save_path)
        if self.use_value_head:
            torch.save(self.value_head.state_dict(), value_head_path)
        return model_save_path, opt_save_path, value_head_path

    def load(self, model_save_path, opt_save_path):
        load_checkpoint(self.model, self.optimizer, model_save_path, opt_save_path)

    def log_metrics(self, episode, train_trajectories, all_queries,
                    total_loss, total_kl_div, total_entropy, advantages,
                    val_trajectories=None, val_queries=None):
        if self.accelerator.is_main_process:
            flat_correct_list = []
            for correct_list in train_trajectories:
                flat_correct_list.extend([t['reward'] for t in correct_list])

            log_dict = {
                "episode": episode + 1,
                "Avg Token Length": np.mean(
                    [encoding['length'].tolist()
                     for encoding in self.train_dataset.all_encodings]
                ).tolist(),
                "Avg Reward": np.mean(flat_correct_list).tolist(),
                "Reward Std": np.std(flat_correct_list).tolist(),
                "Loss": total_loss.tolist(),
                "KL Divergence": total_kl_div.tolist(),
                "Entropy": total_entropy.tolist(),
                "Avg Advantage": np.mean(advantages).tolist(),
                "Advantage Std": np.std(advantages).tolist(),
                "Advantage Min": np.min(advantages).tolist(),
                "Advantage Max": np.max(advantages).tolist()
            }

            try:
                import wandb
                import pandas as pd

                advantange_hist = np.histogram(advantages)
                log_dict["Advantage Hist"] = wandb.Histogram(
                    np_histogram=advantange_hist
                )

                reward_hist = np.histogram(flat_correct_list)
                log_dict["Reward Hist"] = wandb.Histogram(
                    np_histogram=reward_hist
                )

                if val_trajectories:
                    flat_val_correct_list = []
                    for correct_list in val_trajectories:
                        flat_val_correct_list.extend(
                            [t['reward'] for t in correct_list]
                        )
                    log_dict["Avg Val Reward"] = np.mean(
                        flat_val_correct_list
                    ).tolist()
                    log_dict["Val Reward Std"] = np.std(
                        flat_val_correct_list
                    ).tolist()

                    wandb_df = []
                    for traj, query in zip(val_trajectories, val_queries):
                        _dict = {'query': query}
                        for kidx, k in enumerate(traj):
                            _dict[f'attempt{kidx+1}'] = k['messages'][-1]['content']
                        wandb_df.append(_dict)

                    log_dict["val_trajectories"] = wandb.Table(
                        dataframe=pd.DataFrame(wandb_df)
                    )
                    wandb.log(log_dict)

                wandb_df = []
                for traj, query in zip(train_trajectories, all_queries):
                    _dict = {'query': query}
                    for kidx, k in enumerate(traj):
                        _dict[f'attempt{kidx+1}'] = k['messages'][-1]['content']
                    wandb_df.append(_dict)

                log_dict["train_trajectories"] = wandb.Table(
                    dataframe=pd.DataFrame(wandb_df)
                )

                wandb.log(log_dict)
            except ImportError:
                pass

        self.accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# SFT training
# ---------------------------------------------------------------------------

def _stratified_split(
    convos: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split conversations into train/val, stratified by category.

    Groups by ``convo['category']``, shuffles each group independently,
    and splits at ``val_ratio`` (at least 1 val per category).
    """
    import random
    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in convos:
        by_cat[c.get('category', 'unknown')].append(c)

    rng = random.Random(seed)
    train_convos: list[dict] = []
    val_convos: list[dict] = []

    for key in sorted(by_cat.keys()):
        group = by_cat[key]
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        split_idx = len(group) - n_val
        train_convos.extend(group[:split_idx])
        val_convos.extend(group[split_idx:])

    return train_convos, val_convos


def run_sft(
    args,
    stage,
    domain_configs: list[tuple[str, list[dict], str | None]],
    tools: list[dict] | None = None,
) -> None:
    """Run supervised fine-tuning for a pipeline stage.

    Args:
        args: Parsed argument namespace (from train_nlu.py) with fields:
            model_name, sft_epochs, sft_lr, max_tokens, batch_size,
            model_save_path, wandb_project, wandb_name, confidence_threshold.
        stage: The pipeline stage to train.
        domain_configs: List of (domain, eval_set, ensemble_results_path) tuples.
        tools: Optional tool manifest (for tool-calling stages).
    """
    import random
    import shutil
    from pathlib import Path
    from typing import Any

    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.utils.sft_data import SFTDataGenerator, SFTExample
    from training.rollouts import build_turn_examples
    from training.eval import evaluate

    accelerator = Accelerator(mixed_precision='fp16')

    # Build stage kwargs
    stage_kwargs: dict[str, Any] = {}
    if tools:
        stage_kwargs['tools'] = tools

    # Per-domain train/val split + SFT example generation
    all_chat_examples = []
    all_val_examples = []

    for domain, eval_set, ens_path in domain_configs:
        train_convos, val_convos = _stratified_split(
            list(eval_set), args.val_ratio, args.seed,
        )
        print(f'  {domain}: stratified split — '
              + ', '.join(
                  f'{cat}: {sum(1 for c in val_convos if c.get("category") == cat)} val'
                  for cat in sorted({c.get("category", "unknown") for c in eval_set})
              ))

        val_examples = build_turn_examples(val_convos, stage, domain, tools, **stage_kwargs)
        all_val_examples.extend(val_examples)

        generator = SFTDataGenerator(stage, domain, **stage_kwargs)
        if ens_path:
            ens_results = SFTDataGenerator.load_jsonl(ens_path)
            examples = generator.generate_from_ensemble_results(
                ens_results, train_convos, args.confidence_threshold,
            )
        else:
            examples = generator.generate_from_gold_labels(train_convos)

        all_chat_examples.extend([ex.to_chat_format() for ex in examples])
        print(f'  {domain}: {len(examples)} train, {len(val_examples)} val')

    chat_examples = all_chat_examples
    val_examples = all_val_examples

    if not chat_examples:
        print('No SFT examples generated. Check eval set and stage configuration.')
        return

    print(f'Total: {len(chat_examples)} SFT examples, {len(val_examples)} val examples')

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
        collate_fn=collate_sft,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    )

    # Optionally wrap with LoRA
    use_lora = getattr(args, 'use_lora', False)
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else "all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
                    'domain': ','.join(d for d, _, _ in domain_configs),
                    'model': args.model_name,
                    'sft_lr': args.sft_lr,
                    'sft_epochs': args.sft_epochs,
                    'num_examples': len(dataset),
                    'use_lora': use_lora,
                    **(dict(lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout) if use_lora else {}),
                },
            )
        except Exception:
            wandb = None
    else:
        wandb = None

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(args.sft_epochs):
        # Pre-update evaluation
        if args.eval_every > 0 and epoch % args.eval_every == 0 and val_examples:
            temp_save_dir = Path(args.model_save_path) / '_eval_ckpt'
            if accelerator.is_main_process:
                temp_save_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                if use_lora:
                    import copy
                    merged = copy.deepcopy(unwrapped).cpu().merge_and_unload()
                    merged.save_pretrained(temp_save_dir)
                    del merged
                else:
                    unwrapped.save_pretrained(temp_save_dir)
                tokenizer.save_pretrained(temp_save_dir)
            accelerator.wait_for_everyone()

            eval_metrics = evaluate(
                str(temp_save_dir), val_examples, stage, domain_configs[0][0],
                accelerator, max_tokens=args.max_tokens,
                temperature=args.eval_temperature, seed=args.seed,
            )

            if accelerator.is_main_process and wandb is not None:
                wandb.log({**eval_metrics, 'epoch': epoch, 'global_step': global_step})
            if accelerator.is_main_process:
                shutil.rmtree(temp_save_dir, ignore_errors=True)

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
        if use_lora:
            # Save adapter weights
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f'LoRA adapter saved to {save_dir}')
            # Also save a merged version for easy sglang inference
            merged_dir = Path(f'{args.model_save_path}_merged')
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged = unwrapped.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f'Merged model saved to {merged_dir}')
        else:
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f'Model saved to {save_dir}')

        if wandb is not None:
            wandb.finish()
