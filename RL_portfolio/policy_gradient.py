from __future__ import annotations

import torch
import numpy as np


class PolicyGradient:
    def __init__(
        self,
        env_class,
        policy_net,
        optimizer,
        buffer,
        memory,
        batch_size: int = 64,
        device: str = "cpu",
        reward_scaling: float = 1.0,
        verbose: bool = True,
        rebalancing_period: int = 75,
        cost_c_plus=None,
        cost_c_minus=None,
        cost_delta_plus=None,
        cost_delta_minus=None,
    ):
        self.env_class = env_class
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.buffer = buffer
        self.memory = memory
        self.batch_size = batch_size
        self.device = device
        self.reward_scaling = reward_scaling
        self.verbose = verbose
        self.rebalancing_period = rebalancing_period
        self.cost_c_plus = cost_c_plus
        self.cost_c_minus = cost_c_minus
        self.cost_delta_plus = cost_delta_plus
        self.cost_delta_minus = cost_delta_minus

    def train(
        self,
        df: np.ndarray,
        initial_amount: float,
        episodes: int = 100,
        callback=None,
        **env_kwargs,
    ) -> None:
        for episode in range(episodes):
            if self.verbose:
                print(f"\n🧠 Episodio {episode + 1}/{episodes}")

            env = self.env_class(
                df=df,
                initial_amount=initial_amount,
                reward_scaling=self.reward_scaling,
                cost_c_plus=self.cost_c_plus,
                cost_c_minus=self.cost_c_minus,
                cost_delta_plus=self.cost_delta_plus,
                cost_delta_minus=self.cost_delta_minus,
                **env_kwargs,
            )

            state, info = env.reset()
            done = False
            total_reward = 0
            last_action = torch.tensor(self.memory.retrieve()).unsqueeze(0).to(self.device)

            t = 0
            while not done:
                t += 1
                observation = torch.tensor(
                    state["state"] if isinstance(state, dict) else state
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action = self.policy_net(observation, last_action)

                if t % self.rebalancing_period == 0:
                    action_np = action.squeeze().cpu().numpy()
                    next_state, reward, done, _, info = env.step(action_np)

                    # Log dettagliato ad ogni azione
                    print(f"[EP {episode + 1} | STEP {t}]")
                    print(f"  ↪️ Azione: {np.round(action_np, 3)}")
                    print(f"  💰 Reward step: {reward:.6f}")
                    if "transaction_cost" in info:
                        print(f"  💸 Costo transazione: {info['transaction_cost']:.6f}")
                    if "expected_return" in info:
                        print(f"  🔮 Rendimento atteso: {info['expected_return']:.6f}")

                    self.memory.add(action_np)
                    self.buffer.add((observation, action, reward))

                    total_reward += reward
                    state = next_state
                    last_action = action
                else:
                    state = env._observation
                    done = env._terminal

            if self.verbose:
                print(f"🎯 Reward totale episodio: {total_reward:.4f}")

            self._update_policy()

            if callback:
                callback(env)

    def _update_policy(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        observations, actions, rewards = zip(*batch)

        observations = torch.cat(observations).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)

        log_probs = torch.log(torch.sum(self.policy_net(observations, actions) * actions, dim=1))
        loss = -torch.mean(log_probs * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
