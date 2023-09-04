import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

# device = torch.device("cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

gamma = 0.95
max_episode = 100000
anneal_eps = 5000
batch_size = 32

alpha = 0.5
target_update = 10
minimal_size = 50
total_time = 0
epsilon = 0.01
count = 0
cnt = 0

env = ModifiedCleanupEnv()
config = {}
config["channel"] = env.channel
config["height"] = env.height
config["width"] = env.width - 3
# config["width"] = 15 # cleanup 16*15
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [ActorCritic(config).to(device) for _ in range(4)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=3e-4) for i in range(4)]

wandb.init(project='Empathy', entity='kfq20', name='A2C CUv2', notes='prosocial A2C modify cleanup')

for episode in range(1, max_episode+1):
    log_probs = []
    values = []
    rewards = []
    obs = env.reset()
    all_done = False
    step = 0

    total_reward = np.zeros(4)
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    total_punish_num = 0

    for i in range(4):
        symp_agents[i].init_hidden(1)

    while not all_done:
        actions = []
        avail_actions = []
        avail_next_actions = []
        all_log_prob = []
        all_value = []
        for i in range(4):
            avail_action = env.__actionmask__(i)
            avail_action_index = np.nonzero(avail_action)[0]
            state = torch.flatten(torch.tensor(obs[i], dtype=torch.float)).to(device)
            hidden_state = symp_agents[i].eval_hidden.to(device)
            action_prob, state_value, symp_agents[i].eval_hidden = symp_agents[i](state.unsqueeze(0), hidden_state)
            action_prob[:, avail_action == 0] = -999999
            action_prob = F.softmax(action_prob, dim=-1)
            dist = Categorical(action_prob[0])
            if np.random.random() < epsilon:
                action = torch.tensor(np.random.choice(avail_action_index), dtype=torch.int8).to(device)
            else:
                action = dist.sample()
            actions.append(action.item())
            avail_actions.append(avail_action)
            one_log_prob = dist.log_prob(action)
            all_log_prob.append(one_log_prob)
            all_value.append(state_value)

        next_obs, reward, done, info = env.step(actions)
        collect_waste_num = info[0]
        collect_apple_num = info[1]
        # punish_num = info[2]
        total_collect_waste_num += collect_waste_num
        total_collect_apple_num += collect_apple_num
        # total_punish_num += punish_num
        total_reward += reward

        log_probs.append(all_log_prob)
        values.append(all_value)
        rewards.append(reward)

        for i in range(4):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
        
        obs = next_obs
        all_done = done[-1]
        step += 1

    total_actor_loss = 0
    total_critic_loss = 0
    for i in range(4):
        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t][0] + rewards[t][1] + rewards[t][2] + rewards[t][3] + 0.99 * R
            returns[t] = R
            advantages[t] = R - values[t][i].item()
        
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        actor_loss = 0
        critic_loss = 0

        for log_prob, value, return_, advantage in zip(log_probs, values, returns, advantages):
            actor_loss -= log_prob[i] * advantage
            critic_loss += F.smooth_l1_loss(value[i].squeeze(), return_)

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + critic_loss
        optim_symp[i].zero_grad()
        loss.backward()
        optim_symp[i].step()

    wandb.log({'actor loss': total_actor_loss, 'critic loss': total_critic_loss, 'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'waste_num':total_collect_waste_num, 'apple_num':total_collect_apple_num})