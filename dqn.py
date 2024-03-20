import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random

# Deep Q Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*8*8,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,4)
 
    def forward(self,x):
        x = x.view(-1,1,8,8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,64*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ReplayBuffer:
    # 初始化缓冲区
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
 
    # 将一条经验数据添加到缓冲区中
    def push(self,state,action,reward,next_state,done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state,action,reward,next_state,done))
 
    # 随机从缓冲区抽取batch_size大小的经验数据
    def sample(self,batch_size):
        states,actions,rewards,next_states,dones = zip(*random.sample(self.buffer,batch_size))
        return states,actions,rewards,next_states,dones
 
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size  # 状态空间
        self.action_size = action_size # 动作空间
        self.q_net = NeuralNet()  # 估计动作价值 神经网络
        self.target_q_net = NeuralNet() # 计算目标值 神经网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=0.001)  # 初始化Adam优化器
        self.memory = ReplayBuffer(capacity=10000)  # 经验回放缓冲区
        self.gamma = 0.99 # 折扣因子
        self.epsilon = 1.0 # 探索率
        self.epsilon_decay = 0.99995 # 衰减因子
        self.epsilon_min = 0.01 # 探索率最小值
        self.batch_size = 64  # 经验回放每个批次大小
        self.update_rate = 200 # 网络更新频率
        self.steps = 0 # 总步数
 
    # 探索策略 在给定状态下采取动作
    def get_action(self,state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size) # 随机选择动作
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.q_net(state)
        return torch.argmax(q_values,dim=1).item()
 
    # 将状态转移元组存储到经验回放缓冲区
    def remember(self,state,action,reward,next_state,done):
        self.memory.push(state,action,reward,next_state,done)
 
    # 从经验回放缓冲区抽取一个批次的转移样本
    def relay(self):
        if len(self.memory) < self.batch_size:
            return
 
        # 从回放经验中抽取数据
        states,actions,rewards,next_states,dones = self.memory.sample(self.batch_size)
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones)).long()
 
        q_targets = self.target_q_net(next_states).detach()  # 计算下一状态Q值
        q_targets[dones] = 0.0 # 对于已完成状态 将Q值设置为0
 
        # 计算目标Q值
        q_targets = rewards.unsqueeze(1) + self.gamma * torch.max(q_targets,dim=1)[0].unsqueeze(1)
        q_expected = self.q_net(states).gather(1,actions.unsqueeze(1)) # 计算当前状态Q值
 
        # 计算损失值
        loss = F.mse_loss(q_expected,q_targets)
 
        # 通过反向传播更新神经网络的参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        self.steps += 1
 
        # 隔一定步数 更新目标网络
        if self.steps % self.update_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
 
        # 更新epsilon值 使得探索时间衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
 
    def train(self,env,episodes):
        steps = []
        for episode in range(episodes):
            env.reset(complete=False)
            step = 0
            print("episodes:",episode)
            while True:
                step += 1
                action = self.get_action(env.state)  # 获取动作
                next_state, reward, done = env.step(action)  # 执行动作
                self.remember(env.state,action,reward,next_state,done)
                self.relay()
                env.state = next_state  # 更新地图状态
                if done or step > 200:
                    break
            steps.append(step)
        return steps
 
    def test(self,env):
        step = 0
        while True:
            step += 1
            action = self.get_action(env.state)
            next_state,reward,done = env.step(action)
            env.state = next_state
            if done or step > 1000:
                break
 
    def save(self,path):
        torch.save(self.q_net.state_dict(),path+"/value_model.pt")
        torch.save(self.target_q_net.state_dict(),path+"/target_model.pt")
 
    def load(self,path):
        self.q_net.load_state_dict(torch.load(path+"/value_model.pt"))
        self.target_q_net.load_state_dict(torch.load(path+"/target_model.pt"))