import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
import time

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

is_wandb = False

gamma = 0.95
max_episode = 500000
anneal_eps = 10000
total_time = 0
epsilon = 0.99
min_epsilon = 0.01
anneal_epsilon = (epsilon - min_epsilon) / anneal_eps

env = CoinGame()
config = {}
config["channel"] = env.channel
config["height"] = env.obs_height
config["width"] = env.obs_width
config["player_num"] = env.player_num
config["action_space"] = env.action_space

agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]
lola_agents = [ActorCritic(config).to(device) for _ in range(env.player_num)]
for i in range(env.player_num):
    lola_agents[i].load_state_dict(agents[i].state_dict())

optim = [torch.optim.Adam(agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]
optim_lola = [torch.optim.Adam(lola_agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]

buffer = [ReplayBuffer(capacity=1, id=i) for i in range(env.player_num)]

if is_wandb:
    wandb.init(project='Empathy', entity='kfq20', name='lola coingame')

def evaluate():
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_coin_1to1 = 0
    total_coin_1to2 = 0
    total_coin_2to1 = 0
    total_coin_2to2 = 0
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_sd_num = 0
    last_action = np.zeros((env.player_num, env.action_space))
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    for i in range(env.player_num):
        agents[i].init_hidden(1)
    # print('total_time', total_time)
    while not all_done:
        actions = []
        avail_actions = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        for i in range(env.player_num):
            if i == 0 or i == 1 or i == 2 or i == 3:
                avail_action = env.__actionmask__(i)

                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                # input = torch.cat((state, last_action))
                h_in = agents[i].hx.to(device)
                c_in = agents[i].cx.to(device)
                action_prob, state_value, agents[i].hx, agents[i].cx = agents[i](state.unsqueeze(0), h_in, c_in)
                # print(action_prob)
                # action_prob[:, avail_action == 0] = -999999
                action_prob = F.softmax(action_prob, dim=-1)

                dist = Categorical(action_prob[0])
                action = torch.argmax(action_prob)
                one_log_prob = dist.log_prob(action)
                actions.append(action.item())
                avail_actions.append(avail_action)
                log_probs[i].append(one_log_prob.unsqueeze(0))
                values[i].append(state_value)

        next_obs, reward, done, info = env.step(actions)
        if env.name == 'coingame':
            total_coin_1to1 += info[0][0]
            total_coin_1to2 += info[0][1]
            total_coin_2to1 += info[1][0]
            total_coin_2to2 += info[1][1]
        elif env.name == 'snowdrift':
            total_sd_num += info
        elif env.name == 'staghunt':
            hare_num = info[0]
            stag_num = info[1]
            total_hunt_hare_num += hare_num
            total_hunt_stag_num += stag_num

        # obs = next_obs
        total_reward += reward

        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
    
    if is_wandb:
        wandb.log({'eval reward1':total_reward[0], 
                   'eval reward2':total_reward[1],
                #    'eval reward3':total_reward[2],
                #    'eval reward4':total_reward[3], 
                   'eval total r':sum(total_reward), 
                #    'eval hare num':total_hunt_hare_num,
                #    'eval stag num':total_hunt_stag_num,
                #    'eval sd num':total_sd_num,
                    'eval 1to1 coin':total_coin_1to1,
                    'eval 1to2 coin':total_coin_1to2,
                    'eval 2to1 coin':total_coin_2to1,
                    'eval 2to2 coin':total_coin_2to2,
                })

for ep in range(max_episode):
    if (ep+1) % 50 == 0:
        evaluate()
    o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)]
    obs = env.reset()
    all_done = False
    total_coin_1to1 = 0
    total_coin_1to2 = 0
    total_coin_2to1 = 0
    total_coin_2to2 = 0
    total_hunt_stag_num = 0
    total_hunt_hare_num = 0
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    total_punish_num = 0
    total_sd_num = 0
    total_reward = np.zeros(env.player_num)
    last_action = np.zeros((env.player_num, env.action_space))
    step = 0
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    rewards = []

    for i in range(env.player_num):
        lola_agents[i].load_state_dict(agents[i].state_dict())
        lola_agents[i].init_hidden(1)
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
        for i in range(env.player_num):
            if i == 0 or i == 1 or i == 2 or i == 3:
                avail_action = env.__actionmask__(i)
                avail_action_index = np.nonzero(avail_action)[0]

                state = torch.tensor(obs[i], dtype=torch.float).to(device)
                h_in = lola_agents[i].hx.to(device)
                c_in = lola_agents[i].cx.to(device)
                action_prob, state_value, lola_agents[i].hx, lola_agents[i].cx = lola_agents[i](state.unsqueeze(0), h_in, c_in)
                # print(action_prob)
                # action_prob[:, avail_action == 0] = -999999
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

        next_obs, reward, done, info = env.step(actions)
        if env.name == 'coingame':
            total_coin_1to1 += info[0][0]
            total_coin_1to2 += info[0][1]
            total_coin_2to1 += info[1][0]
            total_coin_2to2 += info[1][1]
        elif env.name == 'snowdrift':
            total_sd_num += info
        elif env.name == 'staghunt':
            hare_num = info[0]
            stag_num = info[1]
            total_hunt_hare_num += hare_num
            total_hunt_stag_num += stag_num
        rewards.append(reward)
        for i in range(env.player_num):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
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
            avail_a[i].append(avail_actions[i])
            avail_a_next[i].append(avail_next_actions[i])
            other_a[i].append(other_action)
            d[i].append(done[i])
            pad[i].append(0.)
            
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
            o[j].append(np.zeros((env.player_num, env.channel, env.obs_height, env.obs_width)))
            a[j].append([0 for _ in range(env.player_num)])
            r[j].append(0)
            o_next[j].append(np.zeros((env.player_num, env.channel, env.obs_height, env.obs_width)))
            avail_a[j].append(np.zeros(env.action_space))
            avail_a_next[j].append(np.zeros(env.action_space))
            other_a[j].append(np.zeros(env.player_num))
            d[j].append(1)
            pad[j].append(1)

    total_actor_loss = 0
    total_critic_loss = 0

    for i in range(env.player_num):
        i_log_probs = torch.cat(log_probs[i], dim=0)
        i_values = torch.cat(values[i], dim=0).squeeze()
        returns = np.zeros(len(rewards)-1)
        advantages = np.zeros(len(rewards)-1)
        R = 0
        reward_batch = torch.tensor(r[i][:this_eps_len], dtype=torch.float)

        # compute returns
        state = torch.tensor(obs[i], dtype=torch.float).to(device)
        h_in = lola_agents[i].hx.to(device)
        c_in = lola_agents[i].cx.to(device)
        _, final_value, _, _ = lola_agents[i](state.unsqueeze(0), h_in, c_in)
        R = final_value
        returns = []
        for step in reversed(range(this_eps_len)):
            R = reward_batch[step] + gamma * R * (1-d[i][step])
            returns.insert(0, R)

        returns = torch.cat(returns).squeeze().detach()
        advantage = returns - i_values[:this_eps_len]
        actor_loss = torch.mean(-i_log_probs[:this_eps_len] * advantage.detach())
        critic_loss = advantage.pow(2).mean()

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + 0.5 * critic_loss
        optim_lola[i].zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lola_agents[i].parameters(), max_norm=1.0)
        optim_lola[i].step()

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
                    # '1to2 infer reward gap':total_infer_reward_gap[0],
                    # '2to1 infer reward gap':total_infer_reward_gap[1],
                    '1to1 coin':total_coin_1to1,
                    '1to2 coin':total_coin_1to2,
                    '2to1 coin':total_coin_2to1,
                    '2to2 coin':total_coin_2to2,
                   'episode':ep,
        })
    '==================================================================='
    '==================================================================='
    'lola'
    for id in range(env.player_num):
        o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)], [[] for _ in range(env.player_num)]
        obs = env.reset()
        all_done = False
        total_reward = np.zeros(env.player_num)
        last_action = np.zeros((env.player_num, env.action_space))
        step = 0
        log_probs = [[] for i in range(env.player_num)]
        values = [[] for i in range(env.player_num)]
        rewards = []
        for i in range(env.player_num):
            agents[i].init_hidden(1)
            lola_agents[i].init_hidden(1)

        while not all_done:
            actions = []
            avail_actions = []
            avail_next_actions = []
            probs = []
            all_log_prob = [[] for i in range(env.player_num)]
            all_value = []
            last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
            for i in range(env.player_num):
                if i == 0 or i == 1 or i == 2 or i == 3:
                    avail_action = env.__actionmask__(i)
                    avail_action_index = np.nonzero(avail_action)[0]

                    state = torch.tensor(obs[i], dtype=torch.float).to(device)
                    if i == id:
                        h_in = agents[i].hx.to(device)
                        c_in = agents[i].cx.to(device)
                        action_prob, state_value, agents[i].hx, agents[i].cx = agents[i](state.unsqueeze(0), h_in, c_in)
                    else:
                        h_in = lola_agents[i].hx.to(device)
                        c_in = lola_agents[i].cx.to(device)
                        action_prob, state_value, lola_agents[i].hx, lola_agents[i].cx = lola_agents[i](state.unsqueeze(0), h_in, c_in)
                    # print(action_prob)
                    # action_prob[:, avail_action == 0] = -999999
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

            next_obs, reward, done, info = env.step(actions)
            rewards.append(reward)
            for i in range(env.player_num):
                avail_next_action = env.__actionmask__(i)
                avail_next_actions.append(avail_next_action)
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
                avail_a[i].append(avail_actions[i])
                avail_a_next[i].append(avail_next_actions[i])
                other_a[i].append(other_action)
                d[i].append(done[i])
                pad[i].append(0.)

            actions = np.array(actions)
            one_hot = np.zeros((len(actions), env.action_space))
            one_hot[np.arange(len(actions)), actions] = 1
            last_action = one_hot
            obs = next_obs     
            all_done = done[-1]
            step += 1

        this_eps_len = step

        for i in range(step, env.final_time):
            for j in range(env.player_num):
                o[j].append(np.zeros((env.player_num, env.channel, env.obs_height, env.obs_width)))
                a[j].append([0 for _ in range(env.player_num)])
                r[j].append(0)
                o_next[j].append(np.zeros((env.player_num, env.channel, env.obs_height, env.obs_width)))
                avail_a[j].append(np.zeros(env.action_space))
                avail_a_next[j].append(np.zeros(env.action_space))
                other_a[j].append(np.zeros(env.player_num))
                d[j].append(1)
                pad[j].append(1)

        for i in range(env.player_num):
            buffer[i].add(o[i], a[i], avail_a[i], avail_a_next[i], other_a[i], r[i], o_next[i], d[i], pad[i])

        total_actor_loss = 0
        total_critic_loss = 0

        i_log_probs = torch.cat(log_probs[id], dim=0)
        i_values = torch.cat(values[id], dim=0).squeeze()

        reward_batch = torch.tensor(r[id][:this_eps_len], dtype=torch.float)
        # compute returns
        state = torch.tensor(obs[id], dtype=torch.float).to(device)
        h_in = agents[id].hx.to(device)
        c_in = agents[id].cx.to(device)
        _, final_value, _, _ = agents[id](state.unsqueeze(0), h_in, c_in)
        R = final_value
        returns = []
        for step in reversed(range(this_eps_len)):
            R = reward_batch[step] + gamma * R * (1-d[id][step])
            returns.insert(0, R)
        returns = torch.cat(returns).squeeze().detach()
        advantage = returns - i_values[:this_eps_len]
        actor_loss = torch.mean(-i_log_probs[:this_eps_len] * advantage.detach())
        critic_loss = advantage.pow(2).mean()
        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + 0.5 * critic_loss
        optim[id].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agents[id].parameters(), max_norm=1.0)
        optim[id].step()

    epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon