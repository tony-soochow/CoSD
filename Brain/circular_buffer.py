import random
from collections import namedtuple
import torch
import numpy as np


class Pro_memory:
    def __init__(self, buffer_size, seed, use_t_obs: bool = False, topk = 3):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        random.seed(self.seed)
        
        self.topk = topk
        self._obs_or_vel = (
            (lambda obs, next_obs: obs)
            if use_t_obs
            else (lambda obs, next_obs: next_obs - obs)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha: float = 0.5

    def add(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear() 

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)

    def get_penalty(self,  t_obs_batch, next_t_obs_batch, batch_size=512):
        query_batch = self._obs_or_vel(t_obs_batch, next_t_obs_batch)
        query_batch = query_batch.view(1, -1)
        rewards = torch.zeros(
            *(len(query_batch), 1), dtype=torch.float32, device=self.device
        )
        if len(self.buffer) > batch_size:
            sampled_trajectories = self.sample(batch_size)
            sampled_trajectories = torch.from_numpy(np.vstack(sampled_trajectories)).to(self.device)
            consistency_penalty = compute_penalty(query_batch , sampled_trajectories, topk=self.topk)          
        else:
            consistency_penalty = torch.zeros_like(rewards)
        normalized_penalty = (-self.alpha * consistency_penalty)

        return normalized_penalty


def compute_penalty(
    query_batch: torch.Tensor, buffer_sample_batch: torch.Tensor, topk: int = 3
) -> torch.Tensor:
    """
    Computes the entropy of the query batch with respect to the buffer sample batch.
    """
    query_batch = query_batch[:, None, :]

    buffer_sample_batch = buffer_sample_batch[None, :, :]
    l2_norm = torch.norm(  
        (query_batch - buffer_sample_batch), p=2, dim=-1
    )  # (batch, buffer)
    topk_entropy, _ = torch.topk(l2_norm, topk, dim=-1, largest=False)
    entropy = topk_entropy[:, -1:]
    return entropy
        
    
