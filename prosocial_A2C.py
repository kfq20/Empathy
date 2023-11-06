import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
import time

# device = torch.device("cpu")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

is_wandb = False

MAX_REWARD = 1
MIN_REWARD = -1

gamma = 0.99
max_episode = 100000
anneal_eps = 5000
batch_size = 128
# similar_factor = 1
alpha = 0.5
update_freq = 20
# target_update = 10
minimal_size = 150
total_time = 0
epsilon = 0.1
# min_epsilon = 0.01
# anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0
beta = 1
count = 0
env_obs_mode = 'complete'
window_height = 5
window_width = 5

env = CoinGame()
config = {}
config["channel"] = env.channel
config["height"] = env.height
# config["width"] = env.width - 3
config["width"] = env.width
# config["width"] = 15 # cleanup 16*15
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]
# target_symp_agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(env.player_num)]

if is_wandb:
    wandb.init(project='Empathy', entity='kfq20', name='prosocial coin')
step_time = 0
a2c_time = 0
offline_time = 0
test_time = 0
    
for ep in range(max_episode):
    # print("eps", ep)
    # if (ep+1)%20 == 0:
    #     evaluate()
    # if ep == 480:
    #     print("warning!!!!!!!!!!!!!!!!!")
    # ep_start = time.time()
    o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)]
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_coin_0to0 = 0
    total_coin_0to1 = 0
    total_coin_1to0 = 0
    total_coin_1to1 = 0
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    total_punish_num = 0
    total_sd_num = 0
    last_action = np.zeros((env.player_num, env.action_space))
    step = 0
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    rewards = []
    for i in range(env.player_num):
        symp_agents[i].init_hidden(1)
    # print('total_time', total_time)
    step_s = time.time()
    while not all_done:
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        all_log_prob = [[] for i in range(env.player_num)]
        all_value = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        # forward_start = time.time()
        for i in range(env.player_num):
            if i == 0 or i == 1 or i == 2 or i == 3:
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]

                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                # input = torch.cat((state, last_action))
                h_in = symp_agents[i].hx.to(device)
                c_in = symp_agents[i].cx.to(device)
                action_prob, state_value, symp_agents[i].hx, symp_agents[i].cx = symp_agents[i](state.unsqueeze(0), h_in, c_in)
                # print(action_prob)
                action_prob[:, avail_action == 0] = -999999
                action_prob = F.softmax(action_prob, dim=-1)

                p = torch.sum(action_prob)
                dist = Categorical(action_prob[0])
                if np.random.random() < epsilon:
                    action = torch.tensor(np.random.choice(avail_action_index), dtype=torch.int8).to(device)
                else:
                    action = dist.sample()
                actions.append(action.item())
                avail_actions.append(avail_action)
                one_log_prob = dist.log_prob(action)
                log_probs[i].append(one_log_prob.unsqueeze(0))
                values[i].append(state_value)

        # forward_end = time.time()
        # forward_time += forward_end - forward_start
        next_obs, reward, done, info = env.step(actions)
        total_coin_0to0 += info[0][0]
        total_coin_0to1 += info[0][1]
        total_coin_1to0 += info[1][0]
        total_coin_1to1 += info[1][1]
        # total_sd_num += info
        # collect_waste_num = info[0]
        # collect_apple_num = info[1]
        # punish_num = info[2]
        # total_collect_waste_num += collect_waste_num
        # total_collect_apple_num += collect_apple_num
        # total_punish_num += punish_num
        # hare_num = info[0]
        # stag_num = info[1]
        # total_hunt_hare_num += hare_num
        # total_hunt_stag_num += stag_num

        # all_log_prob = torch.cat(all_log_prob, dim=0)
        # log_probs.append(all_log_prob)
        # values.append(all_value)
        rewards.append(reward)
        for i in range(env.player_num):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
        # obs = next_obs
        total_reward += reward

        for i in range(env.player_num):
            other_action = np.ones(env.player_num) * -1
            for j in range(env.player_num):
                if j == i:
                    continue
                else:
                    other_action[j] = actions[j]
            o[i].append(obs)
            a[i].append(actions)
            r[i].append(reward[i])
            o_next[i].append(next_obs)
            d[i].append(done[i])
            
            # buffer[i].add(obs, actions[i], last_action, avail_actions[i], avail_next_actions[i], other_action, reward[i], next_obs, done[i])
        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
        step += 1

    this_eps_len = step
    print("eps:", ep, "len:", this_eps_len)

    for i in range(step, env.final_time):
        for j in range(env.player_num):
            o[j].append(np.zeros((env.player_num, env.channel, window_height, window_width)))
            a[j].append([0 for _ in range(env.player_num)])
            r[j].append(0)
            o_next[j].append(np.zeros((env.player_num, env.channel, window_height, window_width)))
            d[j].append(1)

    total_actor_loss = 0
    total_critic_loss = 0
    all_weighted_rewards = []
    total_factor = []
    for i in range(env.player_num):
        i_log_probs = torch.cat(log_probs[i], dim=0)
        i_values = torch.cat(values[i], dim=0).squeeze()
        returns = np.zeros(len(rewards)-1)
        other_reward = [0 for _ in range(env.player_num)]
        factor = [0 for _ in range(env.player_num)]
        weighted_rewards = [0 for _ in range(env.final_time)]
        weighted_rewards = torch.tensor(r[0][:this_eps_len], dtype=torch.float) + torch.tensor(r[1][:this_eps_len], dtype=torch.float)
        # with torch.no_grad():
        #     reward_batch = r[i][:this_eps_len]
        #     weighted_rewards = torch.tensor(reward_batch, dtype=torch.float)
        #     for j in range(env.player_num):
        #         if j == i:
        #             continue
        #         else:
        #             other_reward[j] = torch.tensor(r[j][:this_eps_len], dtype=torch.float)
        #             real_factor = torch.ones((this_eps_len, 1, 1))
        #             total_factor.append(real_factor)
        #             weighted_rewards += real_factor.squeeze() * other_reward[j]
            
        state = torch.tensor(obs[i], dtype=torch.float).to(device)
        h_in = symp_agents[i].hx.to(device)
        c_in = symp_agents[i].cx.to(device)
        _, final_value, _, _ = symp_agents[i](state.unsqueeze(0), h_in, c_in)
        R = final_value
        returns = []
        for step in reversed(range(this_eps_len)):
            R = weighted_rewards[step] + gamma * R * (1-d[i][step])
            returns.insert(0, R)

        returns = torch.cat(returns).squeeze().detach()
        advantage = returns - i_values[:this_eps_len]
        actor_loss = torch.mean(-i_log_probs[:this_eps_len] * advantage.detach())
        critic_loss = advantage.pow(2).mean()

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + 0.5 * critic_loss
        optim_symp[i].zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(symp_agents[i].parameters(), max_norm=1.0)
        optim_symp[i].step()
    # factor1to2 = total_factor[0].mean()
    # factor2to1 = total_factor[1].mean()
    if is_wandb:
        wandb.log({'actor loss': total_actor_loss, 
                   'critic loss': total_critic_loss,
                   'reward_1':total_reward[0],
                   'reward_2':total_reward[1],
                #    'reward_3':total_reward[2],
                #    'reward_4':total_reward[3],
                   'total_reward':sum(total_reward),
                #    'waste_num':total_collect_waste_num,
                #    'apple_num':total_collect_apple_num,
                #    'punish_num':total_punish_num,
                #    'snow_num':total_sd_num,
                #    'stag num':total_hunt_stag_num,
                #    'hare num':total_hunt_hare_num,
                    '0to0 coin':total_coin_0to0,
                    '0to1 coin':total_coin_0to1,
                    '1to0 coin':total_coin_1to0,
                    '1to1 coin':total_coin_1to1,
                   'episode':ep,
                #    'factor_1to2':total_factor[0].mean(), #'factor_1to3':total_factor[0][2]/200, 'factor_1to4':total_factor[0][3]/200,
                #    'factor_2to1':total_factor[1].mean(), #'factor_2to3':total_factor[1][2]/200, 'factor_2to4':total_factor[1][3]/200,
                #    'factor_3to1':total_factor[2][0]/200, 'factor_3to2':total_factor[2][1]/200, 'factor_3to4':total_factor[2][3]/200,
                #    'factor_4to1':total_factor[3][0]/200, 'factor_4to2':total_factor[3][1]/200, 'factor_4to3':total_factor[3][2]/200
                    })
    # if buffer[0].size() > minimal_size and ep % update_freq == 0: