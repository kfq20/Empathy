import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义Dueling DQN网络
class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(DuelingDQN, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        x = self.shared_layers(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# DQN算法
def dqn(env_name, hidden_size, num_epochs, batch_size, learning_rate, gamma, target_update):
    env = gym.make(env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = DuelingDQN(num_inputs, num_actions, hidden_size)
    target_model = DuelingDQN(num_inputs, num_actions, hidden_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    replay_buffer = []
    epsilon = 1.0
    epsilon_decay = 1e-3

    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            q_values = model(torch.FloatTensor(state))
            action = q_values.argmax().item()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > batch_size:
                # 从回放缓冲区中取样进行训练
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                current_q_values = model(states)
                current_action_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                next_q_values = target_model(next_states).detach()
                max_next_action_values = next_q_values.max(1)[0]
                expected_q_values = rewards + gamma * max_next_action_values * (1 - dones)

                loss = criterion(current_action_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新目标网络
                if epoch % target_update == 0:
                    target_model.load_state_dict(model.state_dict())

            state = next_state

        epsilon -= epsilon_decay