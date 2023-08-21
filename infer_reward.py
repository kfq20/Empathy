import torch.nn as nn
import torch

class RewardInfer(nn.Module):
    def __init__(self):
        super(RewardInfer, self).__init__()
        self.conv = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32*8*8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        

    def forward(self, inputs):
        x = torch.from_numpy(inputs)
        x = self.conv(inputs)
        x = self.flat(x)
        x = self.fc2(self.relu(self.fc1(x)))
        reward = self.fc3(self.relu(x))
        return reward
    
model_t = RewardInfer()
optimizer_t = torch.optim.Adam(params=model_t.parameters(), lr=1e-4)
loss_func_t = nn.CrossEntropyLoss()
# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.ocnv
def train_reward_infer(model, optimizer, trajectory, policy, loss_func):
    demo_action = trajectory[1]
    my_action = policy(trajectory[0])
    loss = loss_func(demo_action, my_action)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
