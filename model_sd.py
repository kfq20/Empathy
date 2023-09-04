import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(config["channel"]*config["height"]*config["width"]+config["player_num"]*config["action_space"], 128) # 7*16*15+4*7
        self.rnn = nn.GRUCell(128, 128)
        self.fc2 = nn.Linear(128, config["action_space"])
        self.eval_hidden = None
        # self.target_hidden = None

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, 128)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, 128))
        # self.target_hidden = torch.zeros((episode_num, 128))

class Imagine(nn.Module):
    def __init__(self):
        super(Imagine, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1, out_features=6)
        self.fc2 = nn.Linear(in_features=6, out_features=36)
        self.fc3 = nn.Linear(in_features=36, out_features=1)
        # self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
# class Qnet(nn.Module):
#     def __init__(self):
#         super(Qnet, self).__init__()
#         self.conv = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3)
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(32*6*3, 512)
#         self.fc2 = nn.Linear(512, 6)

#     def forward(self, x):
#         x = self.relu(self.conv(x))
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Qself(nn.Module):
    def __init__(self, config):
        super(Qself, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(config["channel"], 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*2*1+config["player_num"]*config["action_space"], 64),
            nn.ReLU(),
            nn.Linear(64, config["action_space"])
        )

    def forward(self, state, last_action):
        x = self.cnn1(state)
        x = self.cnn2(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, last_action), dim=1)
        x = self.fc(x)
        return x
    
class QselfRNN(nn.Module):
    def __init__(self, config):
        super(QselfRNN, self).__init__()
        self.hidden = None
        self.cnn1 = nn.Sequential(
            nn.Conv2d(config["channel"], 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(32*2*1+config["player_num"]*config["action_space"], 128)
        self.fc2 = nn.Linear(128, config["action_space"])
        # self.fc = nn.Sequential(
        #     nn.Linear(32*2*1+config["player_num"]*config["action_space"], 64),
        #     nn.ReLU(),
        #     nn.Linear(64, config["action_space"])
        # )
        self.rnn = nn.GRUCell(128, 128)
        self.relu = nn.ReLU()

    def forward(self, state, last_action, hidden_state):
        x = self.cnn1(state)
        x = self.cnn2(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, last_action), dim=1)
        h_in = hidden_state.reshape(-1, 128)
        x = self.relu(self.fc1(x))
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
    def init_hidden(self, episode_num):
        self.hidden = torch.zeros((episode_num, 128))

class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(config["channel"]*config["height"]*config["width"], 128)
        self.rnn = nn.GRUCell(128, 128)
        self.actor = nn.Linear(128, config["action_space"])
        self.critic = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.hidden = None

    def forward(self, x, hidden_state):
        x = self.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, 128)
        h = self.rnn(x, h_in)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value, h
    
    def init_hidden(self, episode_num):
        self.hidden = torch.zeros((episode_num, 128))

# class QMix(nn.Module):
#     def __init__(self):
#         super(QMix, self).__init__()
        
#     def forward(self, my_q_value, other_q_value, factor):
#         return my_q_value + torch.sum(torch.stack([factor[i]*other_q_value[i] for i in range(len(factor)) if isinstance(factor[i], torch.Tensor)]), dim=0)