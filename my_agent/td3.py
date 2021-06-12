import abc
import copy
import math
import os
import random
from collections import deque
from typing import List, Tuple, Iterable

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from my_agent.states import GameState, clip_direction
from predators_and_preys_env.env import PredatorsAndPreysEnv

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
TRANSITIONS_PER_ROUND = 10000000

NOISE_CLIP = 0.5
ACT_NOISE = 2
A_LOW, A_HIGH = -1, 1


def seed_everything(env, seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
    env.seed(seed)


def iter_to_torch(arr: Iterable) -> torch.Tensor:
    return torch.tensor(np.array(list(arr)), device=DEVICE, dtype=torch.float)


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Agent:
    @abc.abstractmethod
    def act(self, game_state: GameState):
        pass

    def update(self, *args):
        pass


class ChasingPredatorAgent(Agent):
    def act(self, game_state: GameState):
        action = []
        for predator in game_state.predators:
            closest_prey = None
            for prey in game_state.preys:
                if not prey.is_alive:
                    continue
                if closest_prey is None:
                    closest_prey = prey
                else:
                    if closest_prey.dist(predator) > prey.dist(predator):
                        closest_prey = prey
            if closest_prey is None:
                action.append(0.0)
            else:
                action.append(predator.direction(closest_prey))
        return torch.tensor(action, device=DEVICE, dtype=torch.float)


class FleeingPreyAgent(Agent):
    def act(self, game_state: GameState):
        action = []
        for prey in game_state.preys:
            closest_predator = None
            for predator in game_state.predators:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if closest_predator.dist(prey) > prey.dist(predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.0)
            else:
                action.append(closest_predator.direction(prey))
        return torch.tensor(action, device=DEVICE, dtype=torch.float)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, is_predator: bool):
        super().__init__()
        self.baseline = ChasingPredatorAgent() if is_predator else FleeingPreyAgent()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, states: List[GameState]):
        batch = torch.tensor([state.observation for state in states], device=DEVICE, dtype=torch.float)
        res = self.model(batch)
        baseline_pred = torch.stack(tuple(self.baseline.act(state) for state in states), dim=0)
        # print(f"Model: {res}; baseline: {baseline_pred}")
        res = torch.clamp(res + baseline_pred, -1, 1)
        return res


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim, is_predator: bool):
        self.is_predator = is_predator
        self.actor = Actor(state_dim, action_dim, is_predator).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.target_actor.requires_grad_(False)
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)

        self.replay_buffer = deque(maxlen=200000)

    def reset_buffer(self) -> None:
        self.replay_buffer.clear()

    def update(self, state: GameState, action, next_state: GameState, reward: float, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            self._take_step()

    def _take_step(self):
        # Sample batch
        transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in range(BATCH_SIZE)]
        states, actions, next_states, rewards, dones = zip(*transitions)

        state_batch = torch.tensor(np.array([state.observation for state in states]), device=DEVICE, dtype=torch.float)
        action_batch = torch.tensor(np.array(actions), device=DEVICE, dtype=torch.float)
        next_state_batch = torch.tensor(
            np.array([state.observation for state in next_states]), device=DEVICE, dtype=torch.float
        )
        reward_batch = torch.tensor(np.array(rewards), device=DEVICE, dtype=torch.float)
        done_batch = torch.tensor(np.array(dones), device=DEVICE, dtype=torch.float)

        # Update critic
        next_action = self.target_actor(next_states)
        noise = torch.clip(torch.randn(next_action.size()) * ACT_NOISE, min=-NOISE_CLIP, max=NOISE_CLIP)
        next_action = torch.clip(
            next_action + noise,
            min=A_LOW,
            max=A_HIGH,
        )

        next_reward = torch.minimum(
            self.target_critic_1(next_state_batch, next_action), self.target_critic_2(next_state_batch, next_action)
        )
        target_reward = reward_batch + GAMMA * (1 - done_batch) * next_reward

        actual_reward_1 = self.critic_1(state_batch, action_batch)
        actual_reward_2 = self.critic_2(state_batch, action_batch)

        critic_loss_1 = F.mse_loss(actual_reward_1, target_reward)
        critic_loss_2 = F.mse_loss(actual_reward_2, target_reward)

        self.critic_1_optim.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optim.step()

        # Update actor
        self.critic_1.requires_grad_(False)

        pred_action = self.actor(states)
        actor_loss = -self.critic_1(state_batch, pred_action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_1.requires_grad_(True)

        soft_update(self.target_critic_1, self.critic_1)
        soft_update(self.target_critic_2, self.critic_2)
        soft_update(self.target_actor, self.actor)

    def act(self, state: GameState):
        with torch.no_grad():
            return self.actor([state]).squeeze(0).cpu().numpy()

    def save(self, name: str):
        torch.save(self.actor, f"{name}.bin")


def evaluate_policy(predator_agent, prey_agent, seed=0, episodes=5, render: bool = False) -> Tuple[float, float]:
    env = PredatorsAndPreysEnv(render=render)
    env.seed(seed)

    alives = []
    killed = []
    for _ in range(episodes):
        state_dict = GameState(env.reset())
        for _ in range(500):
            state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
            state_dict = GameState(state_dict)
            if done:
                break
        alives.append(sum(prey.is_alive for prey in state_dict.preys))
        killed.append(len(state_dict.preys) - alives[-1])

    return sum(alives) / episodes, sum(killed) / episodes


def train():
    env = PredatorsAndPreysEnv(render=False)
    seed_everything(env)
    state = GameState(env.reset())

    td3_pred = TD3(state_dim=state.observation.shape[0], action_dim=env.predator_action_size, is_predator=True)
    td3_prey = TD3(state_dim=state.observation.shape[0], action_dim=env.prey_action_size, is_predator=False)
    baseline_pred = ChasingPredatorAgent()
    baseline_prey = FleeingPreyAgent()

    eps = 0.1

    for i in tqdm.trange(TRANSITIONS_PER_ROUND):
        pred_action = td3_pred.act(state)
        pred_action = np.clip(pred_action + eps * np.random.randn(*pred_action.shape), -1, +1)

        prey_action = td3_prey.act(state)
        prey_action = np.clip(prey_action + eps * np.random.randn(*prey_action.shape), -1, +1)

        next_state, reward, done = env.step(pred_action, prey_action)
        next_state = GameState(next_state)
        td3_pred.update(state, pred_action, next_state, sum(reward["predators"]), done)
        td3_prey.update(state, prey_action, next_state, sum(reward["preys"]), done)

        state = next_state if not done else GameState(env.reset())

        if i > 20000 and i % 1000 == 0:
            alive, _ = evaluate_policy(baseline_pred, td3_prey, episodes=5)
            _, killed = evaluate_policy(td3_pred, baseline_prey, episodes=5)
            base_alive, base_killed = evaluate_policy(baseline_pred, baseline_prey, episodes=5)
            tqdm.tqdm.write(
                f"Alive score: {alive}, killed score: {killed}" f"Base alive: {base_alive}, base killed: {base_killed}"
            )


if __name__ == "__main__":
    train()
