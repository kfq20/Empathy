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
selfish_agents = [ACself(config).to(device) for _ in range(env.player_num)]

agents_imagine = [Imagine().to(device) for _ in range(env.player_num)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(env.player_num)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(env.player_num)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(env.player_num)]

def counterfactual_factor(my_id, other_id, agents, state, last_action, my_action, real_q_value, observable):
    batch_size = real_q_value.shape[0]
    counterfactual_result = []
    log = []
    counterfactual_result.append(real_q_value)
    # log.append(real_q_value.item())
    other_action = copy.deepcopy(last_action[:, env.action_space*other_id:env.action_space*(other_id+1)])
    virtual_action = torch.nonzero(other_action == 0).view(batch_size, env.action_space-1, 2)
    other_action.fill_(0)
    for i in range(env.action_space-1):
        cur_try_action_pos = virtual_action[:, i, :] # 32*2
        other_action[cur_try_action_pos[:, 0], cur_try_action_pos[:, 1]] = 1 # try another action
        last_action[:, env.action_space*other_id:env.action_space*(other_id+1)] = other_action
        # cf_input = torch.cat((state, last_action), dim=1)
        _, cf_q_value = agents[my_id](state, last_action)
        counterfactual_result.append(cf_q_value)
        # log.append(cf_q_value.item())
        other_action.fill_(0)
    if_log = random.random()
    # if if_log < 0.001:
    #     print("other action", virtual_action)
    #     print(log)
    cf_result = torch.transpose(torch.stack(counterfactual_result), 0, 1)
    min_value, _ = cf_result.min(dim=1)
    max_value, _ = cf_result.max(dim=1)
    factor = 2 * (real_q_value - min_value) / (max_value - min_value) - 1
    return factor.detach().cpu()

def get_loss(batch, id, eps):
    obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]
    episode_num = obs.shape[0]
    self_actor_loss = 0
    self_critic_loss = 0
    symp_loss = 0
    imagine_loss = 0

    # random_start = random.randint(0, env.final_time-51)
    for transition_idx in range(env.final_time):
        obs_transition, next_obs_transition, action_transition, avail_action_transition = obs[:, transition_idx, :, :, :, :], next_obs[:, transition_idx, :, :, :, :], action[:, transition_idx, :], avail_action[:, transition_idx]
        # avail_next_action_transition = avail_next_action[:, transition_idx]
        reward_transition = torch.tensor(reward[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        done_transition = torch.tensor(done[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        pad_transition = torch.tensor(pad[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        other_action_transition = torch.tensor(other_action[:, transition_idx, :], dtype=torch.float).to(device)
        obs_i = obs_transition[:, id, :, :, :]
        done_player = np.zeros((batch_size, env.player_num))
        for j in range(env.player_num):
            if j == id:
                continue
            else:
                done_player[:, j] = np.sum(obs_i[:, j, :, :], axis=(1, 2))
        next_obs_i = next_obs_transition[:, id, :, :, :]
        action_i = torch.tensor(action_transition[:, id]).view(-1, 1).to(device)
        # inputs_next.append(next_obs_transition)
        if transition_idx == 0:
            last_action = np.zeros((episode_num, env.player_num, env.action_space))
        else:
            last_action = action[:, transition_idx - 1]
            last_action = np.eye(env.action_space)[last_action] # one-hot
        
        # if transition_idx == env.final_time - 1:
        #     next_action = np.zeros((episode_num, env.player_num, 6))
        # else:
        #     next_action = action[:, transition_idx + 1]
        #     next_action = np.eye(6)[next_action]

        this_action = action[:, transition_idx]
        this_action = np.eye(env.action_space)[this_action]

        state = torch.tensor(obs_i, dtype=torch.float).to(device)
        next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device)
        last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(episode_num, -1)
        # next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, env.player_num*6)
        this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(episode_num, -1)

        # selfish_inputs = torch.cat((state, last_action), dim=1)
        # selfish_next_inputs = torch.cat((next_state, this_action), dim=1)
        logits, value = selfish_agents[id](state, last_action)
        next_logits, next_value = selfish_agents[id](next_state, this_action)
        log_probs = torch.log(torch.softmax(logits, dim=1).gather(1, action_i)+1e-10)
        # log_probs = torch.log(torch.softmax(logits, dim=1).gather(1, action_i))
        if torch.isnan(log_probs).int().sum() > 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!warning!!!!!!!!!!!!!!!!!")
            print("log is wrong!!!!")
        td_target = reward_transition + 0.99 * next_value * (1 - done_transition
                                                                       )
        td_delta = td_target - value
        self_actor_loss += torch.mean(-log_probs * td_delta.detach())
        self_critic_loss += torch.mean(
            F.mse_loss(value, td_target.detach()))

        # factor = [0 for _ in range(env.player_num)]
        imagine_loss_1 = [0 for _ in range(env.player_num)]
        imagine_loss_2 = [0 for _ in range(env.player_num)]
        j_imagine_obs = [0 for _ in range(env.player_num)]
        j_next_imagine_obs = [0 for _ in range(env.player_num)]
        j_action = [0 for _ in range(env.player_num)]
        j_q = [0 for _ in range(env.player_num)]
        agents_clone = [ACself(config).to(device) for _ in range(env.player_num)]

        for j in range(env.player_num):
            if j == id:
                continue
            else:
                obs_j = torch.tensor(obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                next_obs_j = torch.tensor(next_obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                j_imagine_obs[j] = agents_imagine[id](obs_j).view(batch_size, env.channel, window_height, window_width)
                j_next_imagine_obs[j] = agents_imagine[id](next_obs_j).view(batch_size, env.channel, window_height, window_width)
                imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-obs_j))
                j_action[j] = other_action_transition[:, j].to(torch.int64)
                j_action_onehot = F.one_hot(j_action[j], num_classes=env.action_space).squeeze()
                agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                # j_imagine_obs[j] = j_imagine_obs[j].view(episode_num, -1)
                # j_inputs = torch.cat((j_imagine_obs[j], this_action), dim=1)
                j_q[j], _ = agents_clone[j](j_imagine_obs[j], last_action)
                j_pi = F.softmax(j_q[j], dim=1)
                imagine_loss_1[j] = F.cross_entropy(j_pi, j_action_onehot.to(torch.float))
        imagine_loss += (1 - delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)

    return self_actor_loss, self_critic_loss, symp_loss, imagine_loss

if is_wandb:
    wandb.init(project='Empathy', entity='kfq20', name='prosocial coin')
step_time = 0
a2c_time = 0
offline_time = 0
test_time = 0

def evaluate():
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_collect_waste_num = 0
    total_collect_apple_num = 0
    total_punish_num = 0
    last_action = np.zeros((env.player_num, env.action_space))
    log_probs = [[] for i in range(env.player_num)]
    values = [[] for i in range(env.player_num)]
    for i in range(env.player_num):
        symp_agents[i].init_hidden(1)
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
                h_in = symp_agents[i].hx.to(device)
                c_in = symp_agents[i].cx.to(device)
                action_prob, state_value, symp_agents[i].hx, symp_agents[i].cx = symp_agents[i](state.unsqueeze(0), h_in, c_in)
                # print(action_prob)
                action_prob[:, avail_action == 0] = -999999
                action_prob = F.softmax(action_prob, dim=-1)

                dist = Categorical(action_prob[0])
                action = dist.sample()
                actions.append(action.item())
                avail_actions.append(avail_action)
                one_log_prob = dist.log_prob(action)
                log_probs[i].append(one_log_prob.unsqueeze(0))
                values[i].append(state_value)

        next_obs, reward, done, info = env.step(actions)
        collect_waste_num = info[0]
        collect_apple_num = info[1]
        punish_num = info[2]
        total_collect_waste_num += collect_waste_num
        total_collect_apple_num += collect_apple_num
        total_punish_num += punish_num

        # obs = next_obs
        total_reward += reward

        actions = np.array(actions)
        one_hot = np.zeros((len(actions), env.action_space))
        one_hot[np.arange(len(actions)), actions] = 1
        last_action = one_hot
        obs = next_obs     
        all_done = done[-1]
    
    wandb.log({'eval reward1':total_reward[0], 'eval reward2':total_reward[1],'eval reward3':total_reward[2],'eval reward4':total_reward[3], 'eval total r':sum(total_reward)})

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

            # elif i == 2: # clean waste
            #     avail_action = env.__actionmask__(i)
            #     avail_action_index = np.nonzero(avail_action)[0]
            #     player_i_pos = np.argwhere(obs[i][i] == 1)[0]
            #     if np.sum(obs[i][-2]) == 0:
            #         action = np.random.choice(avail_action_index)
            #     else:
            #         waste_pos = np.argwhere(obs[i][-2] == 1)
            #         distances = np.sum(np.abs(waste_pos - player_i_pos), axis=1)
            #         nearest_waste_pos = waste_pos[np.argmin(distances)]
            #         if nearest_waste_pos[0] < player_i_pos[0]:
            #             action = 0  # 上
            #         elif nearest_waste_pos[0] > player_i_pos[0]:
            #             action = 1  # 下
            #         elif nearest_waste_pos[1] < player_i_pos[1]:
            #             action = 2  # 左
            #         elif nearest_waste_pos[1] > player_i_pos[1]:
            #             action = 3  # 右
            #         else:
            #             action = 5  # clean waste
            #     if avail_action[action] == 0: # not avail, just stay
            #         action = 4
            #     actions.append(action)
            #     avail_actions.append(avail_action)

            # else: # collect apple
            #     avail_action = env.__actionmask__(i)
            #     avail_action_index = np.nonzero(avail_action)[0]
            #     player_i_pos = np.argwhere(obs[i][i] == 1)[0]
            #     if np.sum(obs[i][-1]) == 0:
            #         action = np.random.choice(avail_action_index)
            #     else:
            #         waste_pos = np.argwhere(obs[i][-1] == 1)
            #         distances = np.sum(np.abs(waste_pos - player_i_pos), axis=1)
            #         nearest_waste_pos = waste_pos[np.argmin(distances)]
            #         if nearest_waste_pos[0] < player_i_pos[0]:
            #             action = 0  # 上
            #         elif nearest_waste_pos[0] > player_i_pos[0]:
            #             action = 1  # 下
            #         elif nearest_waste_pos[1] < player_i_pos[1]:
            #             action = 2  # 左
            #         elif nearest_waste_pos[1] > player_i_pos[1]:
            #             action = 3  # 右

            #         else:
            #             action = 6  # collect apple
            #     if avail_action[action] == 0: # not avail, just stay
            #         action = 4
            #     actions.append(action)
            #     avail_actions.append(avail_action)

            # log_probs[i].append()

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
            avail_a[i].append(avail_actions[i])
            avail_a_next[i].append(avail_next_actions[i])
            other_a[i].append(other_action)
            d[i].append(done[i])
            pad[i].append(0.)
            
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
            avail_a[j].append(np.zeros(env.action_space))
            avail_a_next[j].append(np.zeros(env.action_space))
            other_a[j].append(np.zeros(env.player_num))
            d[j].append(1)
            pad[j].append(1)

    for i in range(env.player_num):
        buffer[i].add(o[i], a[i], avail_a[i], avail_a_next[i], other_a[i], r[i], o_next[i], d[i], pad[i])

    total_actor_loss = 0
    total_critic_loss = 0
    # log_probs = torch.cat(log_probs, dim=0)
    # values = torch.cat(values, dim=0)
    ac_s = time.time()
    # total_factor = [[0 for _ in range(env.player_num)] for _ in range(env.player_num)]
    all_weighted_rewards = []
    total_factor = []
    for i in range(env.player_num):
        i_log_probs = torch.cat(log_probs[i], dim=0)
        i_values = torch.cat(values[i], dim=0).squeeze()
        returns = np.zeros(len(rewards)-1)
        advantages = np.zeros(len(rewards)-1)
        R = 0
        other_reward = [0 for _ in range(env.player_num)]
        other_reward_log = [0 for _ in range(env.player_num)]
        factor_log = [0 for _ in range(env.player_num)]
        j_imagine_obs = [0 for _ in range(env.player_num)]
        j_next_imagine_obs = [0 for _ in range(env.player_num)]
        # agents_clone = [ACself(config).to(device) for _ in range(env.player_num)]
        # agents_target_clone = [ACself(config).to(device) for _ in range(env.player_num)]
        j_action = [0 for _ in range(env.player_num)]
        j_q = [0 for _ in range(env.player_num)]
        other_q_value = [0 for _ in range(env.player_num)]
        other_q_target = [0 for _ in range(env.player_num)]
        factor = [0 for _ in range(env.player_num)]
        last_factor = [0 for _ in range(env.player_num)]
        weighted_rewards = [0 for _ in range(env.final_time)]

        with torch.no_grad():
            last_action = np.zeros((1, env.player_num, env.action_space))
            x = np.array(a[i][:this_eps_len-1])
            i_last_action = [sublist[0] for sublist in x]
            i_last_action = torch.tensor(i_last_action).view(this_eps_len-1, 1)
            one_hot_array = np.zeros((x.shape[0], x.shape[1], env.action_space))
            one_hot_array[np.arange(x.shape[0])[:, np.newaxis], np.arange(x.shape[1]), x] = 1
            last_action = one_hot_array
            obs_batch = np.array(o[i][:this_eps_len])
            reward_batch = r[i][:this_eps_len]
            i_obs_batch = obs_batch[1:, i, :,:,:]
            state = torch.tensor(i_obs_batch, dtype=torch.float).to(device)
            last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(last_action.shape[0], -1)

            # _, self_q_value = selfish_agents[i](state, last_action)
            # weighted_rewards = torch.tensor(reward_batch, dtype=torch.float)
            for j in range(env.player_num):
                if j == i:
                    continue
                else:
                    other_reward[j] = torch.tensor(r[j][:this_eps_len], dtype=torch.float)

            #         start_factor = torch.ones((1,1))
            #         origin_factor = counterfactual_factor(i, j, selfish_agents, state, last_action, i_last_action, self_q_value, 0) # len(eps)-1
            #         origin_factor = torch.cat((start_factor, origin_factor))
            #         real_factor = [torch.ones((1, 1))]
            #         last_factor = torch.ones((1, 1))
            #         for t in range(1, this_eps_len):
            #             if t % 4 == 0:
            #                 fluent_factor = 0.8*last_factor + 0.2*sum(origin_factor[t-3:t+1])/4
            #                 real_factor.append(fluent_factor)
            #                 last_factor = fluent_factor
            #             else:
            #                 real_factor.append(last_factor)
            #         real_factor = torch.stack(real_factor)

                    real_factor = torch.ones((this_eps_len, 1, 1))
                    total_factor.append(real_factor)
                    weighted_rewards += real_factor.squeeze() * other_reward[j]
                


        # with torch.no_grad(): # compute weighted total reward
        #     t_factor = [[[] for _ in range(env.player_num)] for _ in range(env.player_num)]
        #     for t in range(len(rewards)):
        #         if t == 0:
        #             last_action = np.zeros((1, env.player_num, env.action_space))
        #             last_action_i = 0
                    
        #         else:
        #             last_action = a[i][t - 1]
        #             last_action = np.eye(env.action_space)[last_action] # one-hot
        #             last_action_i = a[i][t-1][i]

        #         obs_transition = o[i][t]
        #         next_obs_transition = o_next[i][t]
        #         action_transition = a[i][t]
        #         reward_transition = r[i][t]
        #         obs_i = obs_transition[i]
        #         next_obs_i = next_obs_transition[i]
        #         state = torch.tensor(obs_i, dtype=torch.float).to(device).unsqueeze(0)
        #         next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device).unsqueeze(0)

        #         last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(1, -1)
        #         action_i = torch.tensor(action_transition[i]).view(-1, 1).to(device)
        #         last_action_i = torch.tensor(last_action_i).view(-1, 1).to(device)
        #         # selfish_inputs = torch.cat((state, last_action), dim=1)

        #         _, self_q_value = selfish_agents[i](state, last_action)
        #         player_in_view = np.zeros((1, env.player_num))
        #         for j in range(env.player_num):
        #             if j == i:
        #                 continue
        #             else:
        #                 player_in_view[:, j] = np.sum(obs_i[j, :, :], axis=(0, 1))

        #         this_action = a[i][t]
        #         this_action = np.eye(env.action_space)[this_action]
        #         this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(1, -1)
        #         other_action_transition = torch.tensor(other_a[i][t], dtype=torch.int64).to(device)

        #         # get others' reward and weights needed
        #         for j in range(env.player_num):
        #             if j == i:
        #                 continue
        #             else:
        #                 other_reward[j] = r[j][t]

        #                 if t == 0:
        #                     factor[j] = torch.ones((1, 1)).to(device).detach()
        #                     last_factor[j] = torch.ones((1, 1)).to(device).detach()
        #                     factor_log[j] = 0
        #                 else:
        #                     # factor[j] = torch.ones((1, 1)).to(device).detach()
        #                     # t_factor = torch.ones((1, 1)).to(device).detach()
        #                     # t_factor = cleanup_compute_factor(state, state_j, next_state, next_state_j, action_i, action_j, i, j)

        #                     this_t_factor = counterfactual_factor(i, j, selfish_agents, state, last_action, last_action_i, self_q_value, player_in_view[:, j])
        #                     if i == 0:
        #                         print(f'timestep {t} factor to {j}: {this_t_factor.item()}')
        #                     if t % 4 == 0:
        #                         t_factor[i][j].append(this_t_factor)
        #                         ave_factor = sum(t_factor[i][j]) / 4
        #                         factor[j] = 0.8*last_factor[j] + 0.2*ave_factor
        #                         factor_log[j] = factor[j][0].item()
        #                         total_factor[i][j] += factor[j][0].item()
        #                         last_factor[j] = factor[j]
        #                         t_factor[i][j] = []
        #                     else:
        #                         factor[j] = last_factor[j]
        #                         t_factor[i][j].append(this_t_factor)
        #                         factor_log[j] = factor[j][0].item()
        #                         total_factor[i][j] += factor[j][0].item()

        #                     # factor[j] = 0.8*last_factor[j] + 0.2*t_factor
        #                     # factor[j] = t_factor
        #                     # factor[j] += counterfactual_factor(i, j, selfish_agents, state, last_action, action_i, self_q_value, player_in_view[:, j])
        #                     # factor_log[j] = factor[j][0].item()
        #                     # total_factor[i][j] += factor[j][0].item()
        #                     # last_factor[j] = factor[j]
            
        #     # for m in range(4):
        #     #     factor[m] /= len(rewards)
        #         weighted_reward = 0
        #         weighted_reward += reward_transition
        #         # weighted_reward = reward_transition
        #         for k in range(env.player_num):
        #             if k != i:
        #                 # weighted_reward += torch.clamp((1-factor[k]), max=1.0)*reward_transition + factor[k] * other_reward[k]
        #                 weighted_reward += factor[k] * other_reward[k]

        #         weighted_rewards[t] = weighted_reward
        # weighted_rewards = torch.tensor(weighted_rewards, dtype=torch.float).to(device).squeeze()

        # compute returns
        state = torch.tensor(obs[i], dtype=torch.float).to(device)
        h_in = symp_agents[i].hx.to(device)
        c_in = symp_agents[i].cx.to(device)
        _, final_value, _, _ = symp_agents[i](state.unsqueeze(0), h_in, c_in)
        R = final_value
        returns = []
        for step in reversed(range(this_eps_len-1)):
            R = weighted_rewards[step] + gamma * R * (1-d[i][step])
            returns.insert(0, R)

        returns = torch.cat(returns).squeeze().detach()
        advantage = returns - i_values[:this_eps_len-1]
        actor_loss = torch.mean(-i_log_probs[:this_eps_len-1] * advantage.detach())
        critic_loss = advantage.pow(2).mean()

        total_actor_loss += actor_loss
        total_critic_loss += critic_loss
        loss = actor_loss + 0.5 * critic_loss
        optim_symp[i].zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(symp_agents[i].parameters(), max_norm=1.0)
        optim_symp[i].step()
    factor1to2 = total_factor[0].mean()
    factor2to1 = total_factor[1].mean()
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
                   'factor_1to2':total_factor[0].mean(), #'factor_1to3':total_factor[0][2]/200, 'factor_1to4':total_factor[0][3]/200,
                   'factor_2to1':total_factor[1].mean(), #'factor_2to3':total_factor[1][2]/200, 'factor_2to4':total_factor[1][3]/200,
                #    'factor_3to1':total_factor[2][0]/200, 'factor_3to2':total_factor[2][1]/200, 'factor_3to4':total_factor[2][3]/200,
                #    'factor_4to1':total_factor[3][0]/200, 'factor_4to2':total_factor[3][1]/200, 'factor_4to3':total_factor[3][2]/200
                    })
    # if buffer[0].size() > minimal_size and ep % update_freq == 0:
    #     # print(total_reward)
    #     total_self_loss = 0
    #     total_symp_loss = 0
    #     total_imagine_loss = 0
    #     for i in range(env.player_num):
    #         obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = buffer[i].sample(min(buffer[i].size(), batch_size))
    #         episode_num = obs.shape[0]
    #         # symp_agents[i].init_hidden(episode_num)
    #         # # target_symp_agents[i].init_hidden(episode_num)
    #         # symp_agents[i].hidden = symp_agents[i].hidden.to(device)
    #         # target_symp_agents[i].hidden = target_symp_agents[i].hidden.to(device)
    #         batch = [obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad]
    #         # loss_time_start = time.time()
    #         # selfish_agents[i].init_hidden(episode_num)
    #         # target_selfish_agents[i].init_hidden(episode_num)
    #         # selfish_agents[i].hidden = selfish_agents[i].hidden.to(device)
    #         # target_selfish_agents[i].hidden = target_selfish_agents[i].hidden.to(device)
    #         self_actor_loss, self_critic_loss, symp_loss, imagine_loss = get_loss(batch, i, ep)
            
    #         self_loss = 0.5*self_actor_loss + 0.5*self_critic_loss
    #         # symp_loss = torch.mean(symp_loss)
    #         optim_selfish[i].zero_grad()
    #         self_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(selfish_agents[i].parameters(), max_norm=1.0)
    #         optim_selfish[i].step()
    #         total_self_loss += self_loss.item()

    #         # optim_symp[i].zero_grad()
    #         # symp_loss.backward(retain_graph=True)
    #         # optim_symp[i].step()
    #         # total_symp_loss += symp_loss.item()

    #         optim_imagine[i].zero_grad()
    #         imagine_loss.backward()
    #         optim_imagine[i].step()
    #         total_imagine_loss += imagine_loss
    #         # loss_end = time.time()
    #         # loss_time += loss_end - loss_time_start

    # #     count += 1