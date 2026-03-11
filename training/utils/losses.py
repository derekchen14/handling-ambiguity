"""Loss functions and advantage estimation for PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss(reduce=False)


def kl_divergence_batched(p, q):
    ratio = q / p
    return ratio - torch.log(ratio) - 1


def ppo_loss(logits, actions, advantages, eps_high, eps_low, use_kl_div=False,
             old_policy_probs=None, reference_probs=None, label_mask=None,
             beta=.03, entropy_coef=None, use_tok_pg=True):
    probs = logits.softmax(dim=-1)
    sampled_action_probs = torch.gather(probs, 2, actions.unsqueeze(-1)).squeeze(-1)

    # NOTE: The condition below is inverted from the expected PPO formulation
    # (ratio should use old_policy_probs when available). Reproduced as-is for
    # compatibility with the original training code.
    if old_policy_probs is not None:
        ratio = sampled_action_probs / sampled_action_probs.detach()
    else:
        ratio = sampled_action_probs / old_policy_probs.detach()

    clipped_ratio = torch.clamp(ratio, min=1 - eps_low, max=1 + eps_high)
    advantages = advantages.unsqueeze(-1)
    min_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

    if reference_probs is not None:
        kl_div = kl_divergence_batched(sampled_action_probs, reference_probs.detach())
        if use_kl_div:
            min_loss -= beta * kl_div
        kl_div = torch.mean(kl_div).detach()
    else:
        kl_div = None

    entropy = -torch.sum(probs * torch.log(probs), dim=-1)

    if entropy_coef is not None:
        min_loss += entropy_coef * entropy
    entropy = entropy.mean()

    if label_mask is not None:
        if use_tok_pg:
            min_loss = (label_mask * min_loss).sum() / label_mask.sum()
        else:
            min_loss = (label_mask * min_loss).sum(dim=-1) / label_mask.sum(dim=-1)

    return -torch.mean(min_loss), kl_div, entropy


def compute_td_returns(value, rewards, gamma=1.0, lambda_critic=1.0, lambda_actor=1.0):
    """Generalized Advantage Estimation."""
    deltas = rewards.clone()
    prev_value = torch.zeros_like(rewards[:, -1])
    for t in range(rewards.shape[1] - 1, -1, -1):
        deltas[:, t] = deltas[:, t] + gamma * prev_value - value[:, t]
        prev_value = value[:, t]

    advantages_critic = deltas.clone()
    prev_advantage_critic = torch.zeros_like(deltas[:, -1])

    advantages_actor = deltas.clone()
    prev_advantage_actor = torch.zeros_like(deltas[:, -1])

    for t in range(deltas.shape[1] - 1, -1, -1):
        prev_advantage_critic = deltas[:, t] + gamma * lambda_critic * prev_advantage_critic
        prev_advantage_actor = deltas[:, t] + gamma * lambda_actor * prev_advantage_actor

        advantages_critic[:, t] = prev_advantage_critic
        advantages_actor[:, t] = prev_advantage_actor

    return advantages_critic, advantages_actor
