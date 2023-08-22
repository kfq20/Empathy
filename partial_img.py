import torch
from env import *
from model_sd import *
from replay import *
import torch.nn.functional as F
import wandb
import os
import time

# device = torch.device("cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

gamma = 0.9
max_episode = 100000
anneal_eps = 5000
batch_size = 32
# similar_factor = 1
alpha = 0.5
target_update = 10
minimal_size = 50
total_time = 0
epsilon = 0.99
min_epsilon = 0.01
anneal_epsilon = (epsilon - min_epsilon) / anneal_eps
delta = 0
beta = 1
count = 0

cnt = 0

env = SnowDriftEnv()
config = {}
config["channel"] = env.channel
config["height"] = env.height
config["width"] = env.width - 3
config["player_num"] = env.player_num
config["action_space"] = env.action_space

symp_agents = [RNN(config).to(device) for _ in range(4)]
target_symp_agents = [RNN(config).to(device) for _ in range(4)]
selfish_agents = [Qself(config).to(device) for _ in range(4)]
target_selfish_agents = [Qself(config).to(device) for _ in range(4)]
for i in range(4):
    target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
    target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())

agents_imagine = [Imagine().to(device) for _ in range(4)]

optim_symp = [torch.optim.Adam(symp_agents[i].parameters(), lr=1e-4) for i in range(4)]
optim_imagine = [torch.optim.Adam(agents_imagine[i].parameters(), lr=1e-4) for i in range(4)]
optim_selfish = [torch.optim.Adam(selfish_agents[i].parameters(), lr=1e-4) for i in range(4)]
buffer = [ReplayBuffer(capacity=5000, id=i) for i in range(4)]

def get_other_obs(obs, id):
    batch_size = obs.shape[0]
    total_other_obs = []
    total_condition = []
    observable = np.zeros((batch_size, 4))
    for i in range(4):
        if i == id:
            continue
        else:
            for j in range(batch_size):
                if np.sum(obs[j, i]) > 0:
                    observable[j, i] = 1
    other_obs = np.zeros((batch_size, env.player_num, env.channel, env.height, 2*2+1))
    for i in range(4):
        i_obs = []
        if i == id:
            total_other_obs.append(np.zeros((batch_size, env.channel, env.height, 2*2+1)))
            total_condition.append(np.zeros(batch_size))
            continue
        else:
            condition = observable[:, i] == 1
            condition = condition.astype(float)
            total_condition.append(condition)
            for j in range(batch_size):
                if condition[j]:
                    pos = np.where(obs[j][i] > 0)
                    x, y = pos[0][0], pos[1][0]
                    other_obs = obs[j][:, :, max(0, y-2):min(y+2+1, 2*2+1)]
                    if y-2 < 0:
                        other_obs = np.pad(other_obs, ((0, 0), (0, 0), (2-y, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    elif y+2+1 > 5:
                        other_obs = np.pad(other_obs, ((0, 0), (0, 0), (0, y-2)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                    i_obs.append(other_obs)
                else:
                    i_obs.append(np.zeros((env.channel, env.height, 2*2+1)))
            total_other_obs.append(np.array(i_obs))
    return np.array(total_other_obs), np.array(total_condition)

def cleanup_compute_factor(my_state, other_state, my_next_state, other_next_state, my_action, other_action, i, j):
    other_state = other_state.detach().cpu().numpy()
    other_next_state = other_next_state.detach().cpu().numpy()
    my_action = my_action.squeeze().detach().cpu().numpy()
    other_action = other_action.squeeze().detach().cpu().numpy()
    factor = []
    for k in range(32):
        my_value = 0
        other_value = 0
        my_pos = np.argwhere(my_state[k, i, :, :] == 1)
        my_next_pos = np.argwhere(my_next_state[k, i, :, :] == 1)
        other_pos = np.argwhere(other_state[k, j, :, :] == 1)
        other_next_pos = np.argwhere(other_next_state[k, j, :, :] == 1)
        my_rubbish_pos = np.argwhere(my_state[k, -2, :, :] == 1)
        my_apple_pos = np.argwhere(my_state[k, -1, :, :] == 1)
        my_next_rubbish_pos = np.argwhere(my_next_state[k, -2, :, :] == 1)
        my_next_apple_pos = np.argwhere(my_next_state[k, -1, :, :] == 1)
        other_rubbish_pos = np.argwhere(other_state[k, -2, :, :] == 1)
        other_apple_pos = np.argwhere(other_state[k, -1, :, :] == 1)
        other_next_rubbish_pos = np.argwhere(other_next_state[k, -2, :, :] == 1)
        other_next_apple_pos = np.argwhere(other_next_state[k, -1, :, :] == 1)
        my_rubbish_dist = np.sum(np.abs(my_pos - my_rubbish_pos), axis=1)
        my_apple_dist = np.sum(np.abs(my_pos - my_apple_pos), axis=1)
        my_next_rubbish_dist = np.sum(np.abs(my_next_pos - my_next_rubbish_pos), axis=1)
        my_next_apple_dist = np.sum(np.abs(my_next_pos - my_next_apple_pos), axis=1)
        other_rubbish_dist = np.sum(np.abs(other_pos - other_rubbish_pos), axis=1)
        other_apple_dist = np.sum(np.abs(other_pos - other_apple_pos), axis=1)
        other_next_rubbish_dist = np.sum(np.abs(other_next_pos - other_next_rubbish_pos), axis=1)
        other_next_apple_dist = np.sum(np.abs(other_next_pos - other_next_apple_pos), axis=1)
        if my_action[k] == 5:
            if my_state[k, -2, my_pos[0][0], my_pos[0][1]] == 1: # I pick up the rubbish
                my_value = 2
            elif my_state[k, -1, my_pos[0][0], my_pos[0][1]] == 1: # I pick up the apple
                my_value = -2
        else:
            my_value = np.sum(np.exp(-my_next_rubbish_dist)) - np.sum(np.exp(-my_rubbish_dist)) - np.sum(np.exp(-my_next_apple_dist)) + np.sum(np.exp(-my_apple_dist))
        if other_action[k] == 5:
            if other_state[j, -2, other_pos[0][0], other_pos[0][1]] == 1: # other pick up the rubbish
                other_value = 2
            elif other_state[j, -1, other_pos[0][0], other_pos[0][1]] == 1: # other pick up the apple
                other_value = -2
        else:
            other_value = np.sum(np.exp(-other_next_rubbish_dist)) - np.sum(np.exp(-other_rubbish_dist)) - np.sum(np.exp(-other_next_apple_dist)) + np.sum(np.exp(-other_apple_dist))
        factor.append(other_value - my_value)

    return torch.tensor(factor).view(-1, 1).to(device).detach() 

def counterfactual_factor(my_id, other_id, agents, state, last_action, my_action, real_q_value, obsable):
    counterfactual_result = []
    counterfactual_result.append(real_q_value)
    other_action = copy.deepcopy(last_action[:, env.action_space*other_id : env.action_space*(other_id+1)])
    virtual_action = torch.nonzero(other_action == 0).view(batch_size, env.action_space-1, 2)
    other_action.fill_(0)
    for i in range(env.action_space-1):
        cur_try_action_pos = virtual_action[:, i, :] # 32*2
        other_action[cur_try_action_pos[:, 0], cur_try_action_pos[:, 1]] = 1 # try another action
        last_action[:, env.action_space*other_id:env.action_space*(other_id+1)] = other_action
        cf_input = torch.cat((state, last_action), dim=1)
        cf_q_value = agents[my_id](cf_input).gather(1, my_action)
        counterfactual_result.append(cf_q_value)
        other_action.fill_(0)
    cf_result = torch.transpose(torch.stack(counterfactual_result), 0, 1)
    min_value, _ = cf_result.min(dim=1)
    max_value, _ = cf_result.max(dim=1)
    factor = 2 * (real_q_value - min_value) / (max_value - min_value) - 1
    return (factor * obsable.view(-1, 1)).view(-1, 1).to(device).detach()

def snowdrift_compute_factor(my_state, other_state, my_next_state, other_next_state, my_action, other_action, i, j):
    other_state = other_state.detach().cpu().numpy()
    other_next_state = other_next_state.detach().cpu().numpy()
    my_action = my_action.squeeze().detach().cpu().numpy()
    other_action = other_action.squeeze().detach().cpu().numpy()
    factor = []
    for k in range(32):
        my_value = 0
        other_value = 0
        my_pos = np.argwhere(my_state[k, i, :, :] == 1)
        my_next_pos = np.argwhere(my_next_state[k, i, :, :] == 1)
        other_pos = np.argwhere(other_state[k, j, :, :] == 1)
        other_next_pos = np.argwhere(other_next_state[k, j, :, :] == 1)
        my_drift_pos = np.argwhere(my_state[k, 4, :, :] == 1)
        my_next_drift_pos = np.argwhere(my_next_state[k, 4, :, :] == 1)
        other_drift_pos = np.argwhere(other_state[k, 4, :, :] == 1)
        other_next_drift_pos = np.argwhere(other_next_state[k, 4, :, :] == 1)
        my_dist = np.sum(np.abs(my_pos - my_drift_pos), axis=1)
        my_next_dist = np.sum(np.abs(my_next_pos - my_next_drift_pos), axis=1)
        other_dist = np.sum(np.abs(other_pos - other_drift_pos), axis=1)
        other_next_dist = np.sum(np.abs(other_next_pos - other_next_drift_pos), axis=1)
        if my_action[k] == 5:
            my_value = 2
        else:
            my_value = np.sum(np.exp(-my_next_dist)) - np.sum(np.exp(-my_dist))
        if other_action[k] == 5:
            other_value = 2
        else:
            other_value = np.sum(np.exp(-other_next_dist)) - np.sum(np.exp(-other_dist))
        factor.append(other_value - my_value)

    return torch.tensor(factor).view(-1, 1).to(device).detach()

def get_loss(batch, id, eps):
    
    obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]
    episode_num = obs.shape[0]
    self_loss = 0
    symp_loss = 0
    imagine_loss = 0
    
    t11 = 0
    t22 = 0
    t33 = 0
    for transition_idx in range(env.final_time):
        inputs, inputs_next = [], []
        obs_transition, next_obs_transition, action_transition, avail_action_transition = obs[:, transition_idx, :, :, :, :], next_obs[:, transition_idx, :, :, :, :], action[:, transition_idx, :], avail_action[:, transition_idx]
        avail_next_action_transition = avail_next_action[:, transition_idx]
        reward_transition = torch.tensor(reward[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        done_transition = torch.tensor(done[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        pad_transition = torch.tensor(pad[:, transition_idx], dtype=torch.float).view(-1, 1).to(device)
        other_action_transition = torch.tensor(other_action[:, transition_idx, :], dtype=torch.float).to(device)
        obs_i = obs_transition[:, id, :, :, :]
        next_obs_i = next_obs_transition[:, id, :, :, :]
        t1 = time.time()
        other_obs, observable = get_other_obs(obs_i, id)
        next_other_obs, next_observable = get_other_obs(next_obs_i, i)
        t2 = time.time()
        observable = torch.tensor(observable, dtype=torch.float).to(device)
        next_observable = torch.tensor(next_observable, dtype=torch.float).to(device)
        other_obs = torch.tensor(other_obs, dtype=torch.float).to(device) #(4, 32, 7, 8, 5)
        next_other_obs = torch.tensor(next_other_obs, dtype=torch.float).to(device)

        action_i = torch.tensor(action_transition[:, id]).view(-1, 1).to(device)
        # inputs_next.append(next_obs_transition)
        if transition_idx == 0:
            last_action = np.zeros((episode_num, env.player_num, env.action_space))
        else:
            last_action = action[:, transition_idx - 1]
            last_action = np.eye(env.action_space)[last_action] # one-hot
        
        # if transition_idx == env.final_time - 1:
        #     next_action = np.zeros((episode_num, 4, 6))
        # else:
        #     next_action = action[:, transition_idx + 1]
        #     next_action = np.eye(6)[next_action]

        this_action = action[:, transition_idx]
        this_action = np.eye(env.action_space)[this_action]

        state = torch.tensor(obs_i, dtype=torch.float).to(device).view(episode_num, -1)
        next_state = torch.tensor(next_obs_i, dtype=torch.float).to(device).view(episode_num, -1)
        last_action = torch.tensor(last_action, dtype=torch.float).to(device).view(episode_num, -1)
        # next_action = torch.tensor(next_action, dtype=torch.float).to(device).view(episode_num, 4*6)
        this_action = torch.tensor(this_action, dtype=torch.float).to(device).view(episode_num, -1)

        inputs = torch.cat((state, last_action), dim=1)
        inputs_next = torch.cat((next_state, this_action), dim=1)

        selfish_inputs = torch.cat((state, last_action), dim=1)
        selfish_next_inputs = torch.cat((next_state, this_action), dim=1)
        self_q_value = selfish_agents[id](selfish_inputs).gather(1, action_i)
        self_q_target = target_selfish_agents[id](selfish_next_inputs)
        self_q_target[avail_next_action_transition == 0] = -99999
        self_q_target = self_q_target.max(1)[0].view(-1, 1)

        t3 = time.time()

        symp_q_value, symp_agents[id].eval_hidden = symp_agents[id](inputs, symp_agents[id].eval_hidden)
        symp_q_target, target_symp_agents[id].eval_hidden = target_symp_agents[id](inputs_next, target_symp_agents[id].eval_hidden)
        symp_q_value = symp_q_value.gather(1, action_i)
        symp_q_target[avail_next_action_transition == 0] = -99999
        symp_q_target = symp_q_target.max(1)[0].view(-1, 1)

        factor = [0 for _ in range(4)]
        imagine_loss_1 = [0 for _ in range(4)]
        imagine_loss_2 = [0 for _ in range(4)]
        j_imagine_obs = [0 for _ in range(4)]
        j_next_imagine_obs = [0 for _ in range(4)]
        j_action = [0 for _ in range(4)]
        j_q = [0 for _ in range(4)]
        other_q_value = [0 for _ in range(4)]
        other_q_target = [0 for _ in range(4)]
        other_reward = [0 for _ in range(4)]
        agents_clone = [Qself(config).to(device) for _ in range(4)]
        agents_target_clone = [Qself(config).to(device) for _ in range(4)]

        for j in range(4):
            if j == id:
                continue
            else:
                obs_j = other_obs[j]
                next_obs_j = torch.tensor(next_obs_transition[:, j, :, :, :], dtype=torch.float).to(device)
                j_imagine_obs[j] = agents_imagine[id](other_obs[j]).view(batch_size, env.channel, env.height, 2*2+1)
                j_next_imagine_obs[j] = agents_imagine[id](next_other_obs[j]).view(batch_size, env.channel, env.height, 2*2+1)
                imagine_loss_2[j] = torch.mean(torch.abs(j_imagine_obs[j]-obs_j))
                j_action[j] = (other_action_transition[:, j] * observable[j]).to(torch.int64)
                j_action_onehot = F.one_hot(j_action[j], num_classes=env.action_space).squeeze()
                agents_clone[j].load_state_dict(selfish_agents[i].state_dict())
                obs_j = obs_j.view(episode_num, -1)
                j_inputs = torch.cat((obs_j, last_action), dim=1)
                j_q[j] = agents_clone[j](j_inputs)
                j_pi = F.softmax(j_q[j], dim=1)
                if j_pi[observable[j] == 1].shape[0] > 0:
                    imagine_loss_1[j] = F.cross_entropy(j_pi[observable[j] == 1], j_action_onehot.to(torch.float)[observable[j] == 1])

                other_q_value[j] = j_q[j].gather(1, (j_action[j] * observable[j]).to(torch.int64).view(-1, 1)) * observable[j].view(-1, 1).detach()
                
                agents_target_clone[j].load_state_dict(target_selfish_agents[i].state_dict())
                next_obs_j = next_obs_j.view(episode_num, -1)
                next_j_inputs = torch.cat((next_obs_j, this_action), dim=1)
                next_j_q_value = agents_target_clone[j](next_j_inputs)
                other_q_target[j] = next_j_q_value.max(1)[0].view(-1, 1).detach()
                other_reward[j] = (other_q_value[j] - gamma * other_q_target[j]).detach().clamp(min=0)

                if transition_idx == 0:
                    factor[j] = torch.zeros((32, 1)).to(device).detach()
                # factor[j] = snowdrift_compute_factor(obs_i, obs_j, next_obs_i, next_obs_j, action_i, j_action[j], id, j)
                # factor[j] = cleanup_compute_factor(obs_i, obs_j, next_obs_i, next_obs_j, action_i, j_action[j], id, j)
                else:
                    factor[j] = counterfactual_factor(id, j, selfish_agents, state, last_action, action_i, self_q_value, observable[j])

        t4 = time.time()
        self_target = reward_transition + gamma * self_q_target * (1 - done_transition)
        self_td_error = (self_q_value - self_target)
        mask_self_td_error = (1 - pad_transition) * self_td_error
        self_loss += mask_self_td_error ** 2

        target_reward = 0
        for k in range(4):
            if isinstance(factor[k], torch.Tensor):
                target_reward += (1 - factor[k]) * reward_transition + factor[k] * other_reward[k]
        if transition_idx == 20 and eps % 100 == 0 and id == 0:
            print("real obs:", obs_transition[0, 1, :, :, :])
            print("imagine obs:", j_imagine_obs[1][0])
        if transition_idx == 20 and eps % 10 == 0 and id == 0:
            print("eps: ", eps, "===========================")
            print("factor: ",factor)
            print("reward: ", reward_transition)
            print("other reward: ", other_reward)
            print("action: ", action_transition)
        symp_target = target_reward + gamma * symp_q_target * (1 - done_transition)
        symp_td_error = symp_q_value - symp_target
        mask_symp_td_error = (1 - pad_transition) * symp_td_error
        symp_loss += mask_symp_td_error ** 2
        imagine_loss += (1 - delta) * sum(imagine_loss_1) + delta * sum(imagine_loss_2)
        t11 += t2 - t1
        t22 += t3 - t2
        t33 += t4 - t3
    # print("get other loss", t11)
    # print("self forward", t22)
    # print("other forward", t33)

    return self_loss, symp_loss, imagine_loss

wandb.init(project='Empathy', entity='kfq20', name='partial img')
total_step_time = 0
total_get_loss_time = 0
total_backward_time = 0
total_time = 0
total_forward = 0
total_env_step = 0
total_test_time = 0
# total_time = 0
# loss_time = 0
# forward_time = 0
start_time = time.time()
for ep in range(max_episode):
    # ep_start = time.time()
    o, a, other_a, r, o_next, avail_a, avail_a_next, pad, d = [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)]
    obs = env.reset()
    all_done = False
    total_reward = np.zeros(env.player_num)
    total_pick_num = 0
    total_punish_num = 0
    last_action = np.zeros((env.player_num, env.action_space))
    step = 0
    
    for i in range(4):
        symp_agents[i].init_hidden(1)
    # print('total_time', total_time)
    while not all_done:
        
        actions = []
        avail_actions = []
        avail_next_actions = []
        probs = []
        last_action = torch.flatten(torch.tensor(last_action, dtype=torch.float)).to(device)
        # obs = torch.tensor(obs, dtype=torch.float).to(device)
        # forward_start = time.time()
        step_start_time = time.time()
        for i in range(4):
            forward1 = time.time()
            avail_action = env.__actionmask__(i)
            avail_action_index = np.nonzero(avail_action)[0]
            
            state = torch.flatten(torch.tensor(obs[i], dtype=torch.float)).to(device)
            input = torch.cat((state, last_action))
            hidden_state = symp_agents[i].eval_hidden.to(device)
            
            
            q_value, symp_agents[i].eval_hidden = symp_agents[i](input.unsqueeze(0), hidden_state)
            
            forward2 = time.time()
            total_forward += forward2 - forward1
            q_value[:, avail_action == 0] = -float("inf")
            if np.random.random() < epsilon:
                action = np.random.choice(avail_action_index)
            else:
                action = q_value.argmax().item()
            actions.append(action)
            avail_actions.append(avail_action)
            

        # forward_end = time.time()
        # forward_time += forward_end - forward_start
        env_s = time.time()
        next_obs, reward, done, pick_num = env.step(actions)
        if pick_num is not None:
            total_pick_num += pick_num
        env_e = time.time()
        step_end_time = time.time()

        total_env_step += env_e - env_s
        for i in range(4):
            avail_next_action = env.__actionmask__(i)
            avail_next_actions.append(avail_next_action)
        # obs = next_obs
        total_reward += reward
        if len(obs.shape) != 4:
            print(1)
        for i in range(4):
            other_action = np.ones(4) * -1
            for j in range(4):
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
        all_done = done[0]
        step += 1

        total_step_time += step_end_time - step_start_time
    for i in range(step, env.final_time):
        for j in range(4):
            o[j].append(np.zeros((env.player_num, env.channel, env.height, 2*2+1)))
            a[j].append([0,0,0,0])
            r[j].append(0)
            o_next[j].append(np.zeros((env.player_num, env.channel, env.height, 2*2+1)))
            avail_a[j].append(np.zeros(env.action_space))
            avail_a_next[j].append(np.zeros(env.action_space))
            other_a[j].append(np.zeros(4))
            d[j].append(0)
            pad[j].append(1)

    for i in range(4):
        buffer[i].add(o[i], a[i], avail_a[i], avail_a_next[i], other_a[i], r[i], o_next[i], d[i], pad[i])

    
    if buffer[0].size() > minimal_size:
        cnt += 1
        # print(total_reward)
        total_self_loss = 0
        total_symp_loss = 0
        total_imagine_loss = 0
        wandb.log({'reward_1':total_reward[0], 'reward_2':total_reward[1], 'reward_3':total_reward[2], 'reward_4':total_reward[3], 'total_reward':sum(total_reward), 'pick_num':total_pick_num})
        for i in range(4):
            obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad = buffer[i].sample(min(buffer[i].size(), batch_size))
            episode_num = obs.shape[0]
            symp_agents[i].init_hidden(episode_num)
            target_symp_agents[i].init_hidden(episode_num)
            symp_agents[i].eval_hidden = symp_agents[i].eval_hidden.to(device)
            target_symp_agents[i].eval_hidden = target_symp_agents[i].eval_hidden.to(device)
            batch = [obs, action, avail_action, avail_next_action, other_action, reward, next_obs, done, pad]
            # loss_time_start = time.time()
            get_loss_start_time = time.time()
            self_loss, symp_loss, imagine_loss= get_loss(batch, i, ep)
            get_loss_end_time = time.time()
            total_get_loss_time += get_loss_end_time - get_loss_start_time
            
            self_loss = torch.mean(self_loss)
            symp_loss = torch.mean(symp_loss)

            back_s_time = time.time()
            optim_selfish[i].zero_grad()
            self_loss.backward()
            optim_selfish[i].step()
            total_self_loss += self_loss.item()

            optim_symp[i].zero_grad()
            symp_loss.backward(retain_graph=True)
            optim_symp[i].step()
            total_symp_loss += symp_loss.item()

            back_e_time = time.time()
            total_backward_time += back_e_time - back_s_time

            optim_imagine[i].zero_grad()
            imagine_loss.backward()
            optim_imagine[i].step()
            total_imagine_loss += imagine_loss
            loss_end = time.time()
            
        # if cnt % 100 == 0:
        #     print("eps: ", ep, "==========================")
        #     print("my reward:", reward.reshape(reward.shape[0]))
        #     # print("other reward: ", other_reward)
        #     print("my action: ", action)
        #     print("other action", other_action)
        count += 1
        if count % target_update == 0:
            for i in range(4):
                target_selfish_agents[i].load_state_dict(selfish_agents[i].state_dict())
                target_symp_agents[i].load_state_dict(symp_agents[i].state_dict())
        # wandb.log({'self_loss': total_self_loss, 'symp_loss': total_symp_loss, 'imagine_loss':imagine_loss})

    epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon
    end_time = time.time()
    total_time = end_time - start_time
    # print("eps:", ep, "===================================")
    # print("step time", total_step_time)
    # print("get loss time", total_get_loss_time)
    # print("backward time", total_backward_time)
    # print("total time", total_time)
    # print("forward time", total_forward)
    # print("env step time", total_env_step)

    # ep_end = time.time()
    # total_time += ep_end - ep_start
    # print(total_time, loss_time, forward_time)