"""Trajectory cleaning utilities."""

from __future__ import annotations


def clean_trajectories(trajectories):
    for i, datapoint in enumerate(trajectories):
        for j, trajectory in enumerate(datapoint):
            for k, _ in enumerate(trajectory['messages']):
                if not trajectories[i][j]['messages'][k]['content']:
                    trajectories[i][j]['messages'][k]['content'] = ''
    return trajectories


def clean_flattened_trajectories(trajectories):
    for i, trajectory in enumerate(trajectories):
        for j, _ in enumerate(trajectory['messages']):
            if not trajectories[i]['messages'][j]['content']:
                trajectories[i]['messages'][j]['content'] = ''
    return trajectories
