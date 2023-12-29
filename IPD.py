import torch
from env import IPD
from model_sd import IPDModel, IPDActor, IPDCritic
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
import collections
import numpy as np
import random

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

is_wandb = True
gamma = 0.99
max_episode = 100000
epsilon = 0.1
env = IPD()
agents = [IPDActor().to(device) for _ in range(2)]
critics = [IPDCritic().to(device) for _ in range(2)]
optim_actor = [torch.optim.Adam(agents[i].parameters(), lr=1e-4) for i in range(2)]
optim_critic = [torch.optim.Adam(critics[i].parameters(), lr=1e-4) for i in range(2)]
action_buffer = collections.deque(maxlen=100)
# my_buffer = collections.deque(maxlen=100)
# opponent_buffer = collections.deque(maxlen=100)

if is_wandb:
    wandb.init(project='Empathy', entity='kfq20', name='IPD_easy no W')


for ep in range(max_episode):
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(2)
    weights = [[], []]
    log_probs = [[] for i in range(2)]
    values = [[] for i in range(2)]
    rewards = []
    dones = []
    last_action = None
    while not all_done:
        actions = []
        for i in range(2):
            state = torch.tensor(obs, dtype=torch.float).to(device)
            # logits, value = agents[i](state.unsqueeze(0))
            logits = agents[i](state.unsqueeze(0))
            value = critics[i](state.unsqueeze(0))
            action_prob = F.softmax(logits, dim=-1)
            dist = Categorical(action_prob[0])
            if np.random.random() < epsilon:
                action = torch.tensor(np.random.choice(2), dtype=torch.int8).to(device)
            else:
                action = dist.sample()
            actions.append(action.item())
            one_log_prob = dist.log_prob(action)
            log_probs[i].append(one_log_prob.unsqueeze(0))
            values[i].append(value)
        next_obs, reward, done, _ = env.step(actions)
        if last_action is None:
            weights[0].append(1)
            weights[1].append(1)
        else:
            if last_action[0] == 0 and last_action[1] == 0: # CC
                weights[0].append(1)
                weights[1].append(1)
            elif last_action[0] == 0 and last_action[1] == 1: # CD
                weights[0].append(-1)
                weights[1].append(1)
            elif last_action[0] == 1 and last_action[1] == 0: # DC
                weights[0].append(1)
                weights[1].append(-1)
            elif last_action[0] == 1 and last_action[1] == 1: # DD
                weights[0].append(0)
                weights[1].append(0)
        rewards.append(reward)
        dones.append(done)
        total_reward += reward
        action_buffer.append(actions)
        all_done = done[0]
        obs = next_obs
        last_action = actions

    rewards = np.array(rewards)
    total_actor_loss = 0
    total_critic_loss = 0
    action_hist = random.sample(action_buffer, 100)
    action_hist = np.array(action_hist) # 100*2
    cc_num = np.sum((action_hist[:, 0] == 0) & (action_hist[:, 1] == 0))
    cd_num = np.sum((action_hist[:, 0] == 0) & (action_hist[:, 1] == 1))
    dc_num = np.sum((action_hist[:, 0] == 1) & (action_hist[:, 1] == 0))
    dd_num = np.sum((action_hist[:, 0] == 1) & (action_hist[:, 1] == 1))
    # W = [0, 0]
    # W[0] = ((cc_num+dc_num) / len(action_buffer) * 1 - cd_num/len(action_buffer)*1)# *(1-0.98)+0.98*w_last[0]
    # W[1] = ((cc_num+cd_num) / len(action_buffer) * 1 - dc_num/len(action_buffer)*1)# *(1-0.98)+0.98*w_last[1]
    for i in range(2):
        i_log_probs = torch.cat(log_probs[i], dim=0)
        i_values = torch.cat(values[i], dim=0).squeeze()
        my_reward = rewards[:, i]
        other_reward = rewards[:, 1-i]
        # weighted_reward = [1-num for num in weights[i]]*my_reward + weights[i]*other_reward
        weighted_reward = my_reward
        state = torch.tensor(obs, dtype=torch.float).to(device)
        # _, final_value = agents[i](state.unsqueeze(0))
        final_value = critics[i](state.unsqueeze(0))
        R = final_value
        returns = []
        for step in reversed(range(env.final_time)):
            R = weighted_reward[step] + gamma * R * (1-dones[step][i])
            returns.insert(0, R)
        returns = torch.cat(returns).squeeze().detach()
        advantage = returns - i_values
        # print("returns",returns)
        # print("values", i_values)
        actor_loss = torch.mean(-i_log_probs * advantage.detach())
        critic_loss = advantage.pow(2).mean()

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        # if ep <= 10000:
        #     optim_critic[i].zero_grad()
        #     critic_loss.backward()
        #     optim_critic[i].step()
        # else:
        optim_critic[i].zero_grad()
        critic_loss.backward()
        optim_critic[i].step()
        
        optim_actor[i].zero_grad()
        actor_loss.backward()
        optim_actor[i].step()

    # w_last[0] = W[0]
    # w_last[1] = W[1]
    if ep % 100 == 0: # evaluate
        CC_state = np.array([1,0,0,0])
        CC_state = torch.tensor(CC_state, dtype=torch.float).to(device)
        CD_state = np.array([0,1,0,0])
        CD_state = torch.tensor(CD_state, dtype=torch.float).to(device)
        DC_state = np.array([0,0,1,0])
        DC_state = torch.tensor(DC_state, dtype=torch.float).to(device)
        DD_state = np.array([0,0,0,1])
        DD_state = torch.tensor(DD_state, dtype=torch.float).to(device)
        with torch.no_grad():
            CC_logits_0 = agents[0](CC_state.unsqueeze(0))
            CC_value = critics[0](CC_state.unsqueeze(0))
            CD_logits_0 = agents[0](CD_state.unsqueeze(0))
            CD_value = critics[0](CD_state.unsqueeze(0))
            DC_logits_0 = agents[0](DC_state.unsqueeze(0))
            DC_value = critics[0](CC_state.unsqueeze(0))
            DD_logits_0 = agents[0](DD_state.unsqueeze(0))
            DD_value = critics[0](DD_state.unsqueeze(0))
            CC_logits_1 = agents[1](CC_state.unsqueeze(0))
            CC_value_2 = critics[1](CC_state.unsqueeze(0))
            CD_logits_1 = agents[1](CD_state.unsqueeze(0))
            CD_value_2 = critics[1](CD_state.unsqueeze(0))
            DC_logits_1 = agents[1](DC_state.unsqueeze(0))
            DC_value_2 = critics[1](DC_state.unsqueeze(0))
            DD_logits_1 = agents[1](DD_state.unsqueeze(0))
            DD_value_2 = critics[1](DD_state.unsqueeze(0))
            print("agent 1, CC value", CC_value)
            print("agent 1, CD value", CD_value)
            print("agent 1, DC value", DC_value)
            print("agent 1, DD value", DD_value)
            print("agent 2, CC value", CC_value_2)
            print("agent 2, CD value", CD_value_2)
            print("agent 2, DC value", DC_value_2)
            print("agent 2, DD value", DD_value_2)

            CC0_action_prob = F.softmax(CC_logits_0, dim=-1)
            CD0_action_prob = F.softmax(CD_logits_0, dim=-1)
            DC0_action_prob = F.softmax(DC_logits_0, dim=-1)
            DD0_action_prob = F.softmax(DD_logits_0, dim=-1)
            CC1_action_prob = F.softmax(CC_logits_1, dim=-1)
            CD1_action_prob = F.softmax(CD_logits_1, dim=-1)
            DC1_action_prob = F.softmax(DC_logits_1, dim=-1)
            DD1_action_prob = F.softmax(DD_logits_1, dim=-1)
            print()
            if is_wandb:
                wandb.log({
                    'P(C|CC)_agent1':CC0_action_prob[0][0].item(),
                    'P(C|CD)_agent1':CD0_action_prob[0][0].item(),
                    'P(C|DC)_agent1':DC0_action_prob[0][0].item(),
                    'P(C|DD)_agent1':DD0_action_prob[0][0].item(),
                    'P(C|CC)_agent2':CC1_action_prob[0][0].item(),
                    'P(C|CD)_agent2':CD1_action_prob[0][0].item(),
                    'P(C|DC)_agent2':DC1_action_prob[0][0].item(),
                    'P(C|DD)_agent2':DD1_action_prob[0][0].item(),
                })
    if is_wandb:
        wandb.log({'actor loss': total_actor_loss, 
                   'critic loss': total_critic_loss,
                   'reward_1':total_reward[0],
                   'reward_2':total_reward[1],
                   'total_reward':sum(total_reward),
                    'CC':cc_num,
                    'CD':cd_num,
                    'DC':dc_num,
                    'DD':dd_num,
                    })
