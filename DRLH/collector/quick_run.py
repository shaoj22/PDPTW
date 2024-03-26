'''
File: quick_run.py
Project: DRLH
File Created: Saturday, 27th May 2023 12:07:04 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

#%%
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from tianshou.data import Batch, ReplayBuffer, to_torch_as
import torch, numpy as np
import torch.nn as nn
import gym as gym
import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, PGPolicy, DQNPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import datetime
from Env import ALNS_Env

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {
    "hidden_shape" : [256, 256],
    "lr" : 1e-5,
    "gamma" : 0.5,
    "epoch" : 100,
    "log_dir" : "DRLH\\Log\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
}

#%%
# 1. Environment
# create 2 vectorized environments both for training and testing
env = ALNS_Env()
train_envs = DummyVectorEnv([lambda: ALNS_Env() for _ in range(1)])
test_envs = DummyVectorEnv([lambda: ALNS_Env() for _ in range(1)])

#%%
# 2. Model
# net is shared head of the actor and the critic
# rewrite the forward function to realize mask 
class MyNet(Net):
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class MyActor(Actor):
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # logits, hidden = self.preprocess(obs.reshape(len(obs), -1), state)
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits, hidden

class MyCritic(Critic):
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # logits, _ = self.preprocess(obs.reshape(len(obs), -1), state=kwargs.get("state", None))
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)


net = MyNet(env.observation_space.shape, hidden_sizes=args["hidden_shape"], device=device)
actor = MyActor(net, env.action_space.n, device=device).to(device)
critic = MyCritic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)

# optimizer of the actor and the critic
optim = torch.optim.AdamW(actor_critic.parameters(), lr=args["lr"])

#%%
# 3. Policy

distributions = torch.distributions.Categorical
policy = PPOPolicy(actor, critic, optim, distributions, action_space=env.action_space, deterministic_eval=True, discount_factor=args["gamma"])
# deterministic_eval=True means choose best action in evaluation

#%%
# 4. Collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)), exploration_noise=True)
test_collector = Collector(policy, test_envs)

#%%
# 5. Trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
writer = SummaryWriter(log_dir=args["log_dir"])
writer.add_text("args", str(args))
logger = TensorboardLogger(writer)
result = onpolicy_trainer(
    policy, 
    train_collector, 
    test_collector, 
    max_epoch=5, 
    step_per_epoch=1000, 
    repeat_per_collect=10, 
    episode_per_test=10, 
    batch_size=64, 
    step_per_collect=100, 
    stop_fn=lambda mean_reward: mean_reward >= 800, 
    logger=logger,
)

# show result
print(result)

#%%
# watch performance
policy.eval()
result = test_collector.collect(n_episode=1, render=False)
print("final reward: {}, length: {}".format(result["rew"].mean(), result["lens"].mean()))












