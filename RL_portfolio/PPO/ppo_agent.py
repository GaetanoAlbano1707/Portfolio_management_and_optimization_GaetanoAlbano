import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus()  # output > 0 per Dirichlet
        )

    def forward(self, x):
        return self.model(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

class PPOAgent:
    def __init__(self, input_dim, num_assets, lr=1e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01):
        self.num_assets = num_assets
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

        self.policy_net = PolicyNetwork(input_dim, num_assets)
        self.value_net = ValueNetwork(input_dim)

        self.policy = self.policy_net  # ✅ serve per salvarla come agent.policy

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).flatten().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state)
            if torch.isnan(probs).any():
                print("⚠️ NaN nei probs (select_action) - uso distribuzione uniforme")
                probs = torch.ones_like(probs)
        dist = torch.distributions.Dirichlet(probs + 1e-5)
        action = dist.sample().cpu().numpy().flatten()
        return action, probs.cpu().numpy(), state

    def evaluate_action(self, states, actions):
        probs = self.policy_net(states)

        if torch.isnan(probs).any():
            print("⚠️ NaN nei probs in evaluate_action - fallback a uniformi")
            probs = torch.ones_like(probs)

        dist = torch.distributions.Dirichlet(probs + 1e-5)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(states).squeeze(-1)
        return log_probs, values, entropy

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * gae * (1 - dones[i])
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, rewards, dones):
        # --- pulizia input ---
        states_np = np.array(states).reshape(len(states), -1)
        states_np = np.nan_to_num(states_np, nan=0.0, posinf=0.0, neginf=0.0)
        states = torch.FloatTensor(states_np).to(self.device)

        actions = torch.FloatTensor(np.array(actions)).to(self.device)

        with torch.no_grad():
            values = self.value_net(states).squeeze(-1).cpu().numpy()
            values = np.append(values, 0)  # bootstrap
            advantages = self.compute_advantages(rewards, values, dones)
            targets = advantages + values[:-1]

        advantages = torch.FloatTensor(advantages).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)

        for _ in range(5):
            log_probs_old, _, _ = self.evaluate_action(states, actions)
            probs = self.policy_net(states)

            if torch.isnan(probs).any():
                print("⚠️ NaN nei probs nel training - skip batch")
                continue

            dist = torch.distributions.Dirichlet(probs + 1e-5)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = torch.exp(log_probs - log_probs_old.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            values_pred = self.value_net(states).squeeze(-1)
            value_loss = nn.MSELoss()(values_pred, targets)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
