'''
动态路由：结合topk-softmax或者softmax-threshold，由大网络出现功能分区但整体一个网络
功能分区：迭代调度，记忆（时序），输入处理，输出生成
类脑学习：非BP，奖励（多巴胺）引起的权重固化学习，其他不激活的则退化
记忆利用：短期结果和长期知识
可能升维
稀疏冗余
迭代优化
能动探索
强化模式
符号独热
覆盖检测
有限算子：OneHot, Matmul(EvenPureAdd)，Add, Squeeze(Tanh/ReLU), Softmax, Route/Attent,   
'''

import random
class Math:  #强化环境：多行，加法
    def __init__(self, range_min=1, range_max=3, sample_step=1, show_sample=0):  
        self.action_nop = ' '
        self.action_end = ';'
        self.actions = ['0','1','2','3','4','5','6','7','8','9']+[self.action_nop]+[self.action_end]
        self.state_equal = '='
        self.states = self.actions+['+']+[self.state_equal]
        self.width = len(str(range_max-1)+'+'+str(range_max-1)+'='+str((range_max-1)+(range_max-1))+';')
        self.reward_ok_yes = +0.9
        self.reward_ok_nop = +0.2
        self.reward_ok_end = +0.1
        self.reward_no_yes = -0.8
        self.reward_no_nop = -0.2
        self.reward_no_end = -0.1
        self.reward_do_any =  0.0
        self.perfect = self.reward_ok_nop*4 + self.reward_ok_yes + self.reward_ok_end
 
        self.dataset = []  
        i = range_min
        while i < range_max:
            j = range_min
            while j < range_max:
                sample = str(i)+'+'+str(j)+'='+str(i+j)+self.action_end 
                self.dataset.append(sample)          
                j = j + random.randint(1, sample_step)
            i = i + random.randint(1, sample_step) 
        self.free()
        #
        if show_sample:
            for one in self.dataset:
                print(one)
            print()

    def free(self, init_line_random=True):
        self.offsetLine = random.randint(0, len(self.dataset)) if init_line_random else 0
        self.offsetChar = -1
        self.response = ''
        return self.action_nop
    
    def step(self, action):
        def _action():
            if action==self.action_nop:
                self.response = ''
            else:
                self.response += action
               
        def _state():
            if self.offsetLine == len(self.dataset):
                self.offsetLine = 0
            if self.offsetChar == len(self.dataset[self.offsetLine])-1:
                self.offsetLine += 1 
                if self.offsetLine == len(self.dataset):
                    self.offsetLine = 0 
                self.offsetChar = -1          
            self.offsetChar += 1
            state = self.dataset[self.offsetLine][self.offsetChar]
            return state
            
        def _reward():
            equalIndex = self.dataset[self.offsetLine].index(self.state_equal)
            if self.offsetChar > equalIndex:
                answer = self.dataset[self.offsetLine][equalIndex+1:-1]
                if self.response == answer:
                    return self.reward_ok_yes
                else:
                    if self.offsetChar < len(self.dataset[self.offsetLine])-2:      #多位数答案中途，不惩，不奖
                        return self.reward_do_any
                    elif self.offsetChar < len(self.dataset[self.offsetLine])-1:    #多位数答案结束，惩罚
                        return self.reward_no_yes
                    else:                                                           #多位数答案后续
                        if action==self.action_end:                                 #  结束+
                            return self.reward_ok_end
                        else:                                                       #  其他-
                            return self.reward_no_end
            else:
                if action==self.action_nop:
                    return self.reward_ok_nop
                else:
                    return self.reward_no_nop

        _action() 
        state = _state()
        return state, _reward(), state==self.action_end

class Coding:
    def __init__(self, i2t):
        self.i2t = i2t
        self.t2i = {}
        for i,t in enumerate(i2t):
            self.t2i[t] = i
    
    def encode(self, t):
        return self.t2i[t]
    
    def decode(self, i):
        return self.i2t[i]

class Memory:
    def __init__(self, size):
        self.size = size
        self.data = []
        self.text = []

    def add_data_text(self, x_data, x_text):
        if len(self.data) >= self.size:
            self.data = self.data[1:]
            self.text = self.text[1:]
        self.data.append(x_data)
        self.text.append(x_text)

    def get_last_data(self):
        return self.data if len(self.data)==self.size else None

    def get_last_text(self):
        return ''.join(self.text)

class Buffer:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, state, action, state_prime, reward, probability_action):
        self.data.append((reward, probability_action))

    def get_reward_probability(self, env):
        r9 = 0
        r2 = 0
        r1 = 0
        for r,p in self.data:
            if r==env.reward_ok_yes:
                r9 = 1
            elif r==env.reward_ok_nop:
                r2 = 1  
            elif r==env.reward_ok_end:
                r1 = 1
            else:
                pass  
        return self.data if ((r9 and r2) or (r9 and r1) or (r2 and r1)) else []

import torch
import math
class Network(torch.nn.Module):
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, max_length, hidden_dimension, dropout_ratio=0.1):
            super().__init__()
            self.P = torch.zeros((1, max_length, hidden_dimension))
            x = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, hidden_dimension, 2, dtype=torch.float32) / hidden_dimension)
            self.P[:, :, 0::2] = torch.sin(x)
            self.P[:, :, 1::2] = torch.cos(x)
            self.dropout = torch.nn.Dropout(dropout_ratio)

        def forward(self, e):
            p = self.P[: , :e.shape[1] , :]
            m = e + p.to(e.device)
            return self.dropout(m)

    class MultiHeadSelfAttention(torch.nn.Module):
        def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
            super().__init__()
            assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
            self.dim_in = dim_in
            self.dim_k = dim_k
            self.dim_v = dim_v
            self.num_heads = num_heads
            self.linear_q = torch.nn.Linear(dim_in, dim_k, bias=False)
            self.linear_k = torch.nn.Linear(dim_in, dim_k, bias=False)
            self.linear_v = torch.nn.Linear(dim_in, dim_v, bias=False)
            self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

        def forward(self, x):
            batch, n, dim_in = x.shape
            assert dim_in == self.dim_in
            nh = self.num_heads
            dk = self.dim_k // nh
            dv = self.dim_v // nh
            q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
            k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
            v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)
            dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
            dist = torch.softmax(dist, dim=-1)
            att = torch.matmul(dist, v)
            att = att.transpose(1, 2).reshape(batch, n, self.dim_v)
            return att

    def __init__(self, intput_length, input_dimension, output_dimensionm, hidden_dimension=32, backbone_size=2):
        super().__init__()
        self.embeder = torch.nn.Embedding(intput_length*input_dimension, hidden_dimension)
        self.encoder = self.__class__.PositionalEncoding(intput_length, hidden_dimension)
        self.backbone = torch.nn.ModuleList()
        for _ in range(backbone_size):
            self.backbone.append(self.__class__.MultiHeadSelfAttention(hidden_dimension, hidden_dimension, hidden_dimension))
        self.gather = torch.nn.Sequential(torch.nn.Linear(intput_length*hidden_dimension, hidden_dimension), torch.nn.ReLU(), torch.nn.Linear(hidden_dimension, output_dimensionm), torch.nn.Softmax(dim=-1))  #softmax for Categorical
   
    def forward(self, x):
        o = self.embeder(x)
        o = self.encoder(o)
        for layer in self.backbone:
            o = layer(o)
        o = o.reshape(o.shape[0], -1)
        o = self.gather(o)
        return o

class Mathink:
    def __init__(self, env):
        self.env = env
        self.network = Network(self.env.width, len(self.env.states), len(self.env.actions))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.coding = Coding(self.env.states)

    def learn(self, total_learn=30000, reward_decay=0.98):
        memory = Memory(size=self.env.width)
        state = self.env.free()
        for i in range(self.env.width):
            action = self.env.actions[i%len(self.env.actions)]
            state_prime, reward, done = self.env.step(action)
            memory.add_data_text(self.coding.encode(state), state)
            state = state_prime
        #
        state = self.env.free()
        for t in range(total_learn):
            buffer = Buffer(size=32)
            score = 0
            done = False
            while not done:   
                memory.add_data_text(self.coding.encode(state), state)                     #John: 构造好的记忆，要么结构化，要么网络本身
                state_memory = memory.get_last_data()
                state_memory = torch.tensor(state_memory, dtype=torch.long)                #John: 必须包含必要的信息比如各种结构信息，并有合适的网络去抓取
                state_memory = state_memory.view(-1, state_memory.shape[-1])
                probability = self.network(state_memory)[0]                                #John: 在线逐样本学习效率不高
                action = torch.distributions.Categorical(probability).sample().tolist()    #John: 训练的时候，不仅是利用，更需要探索
                if 0:
                    action_deterministic = torch.argmax(probability.detach()).tolist() 
                    if action != action_deterministic:
                        print('explore:', 'action', action, '<-', 'action_deterministic', action_deterministic)
                    else:
                        print('exploit:', 'action', action, '==', 'action_deterministic', action_deterministic)
                action_decoded = self.coding.decode(action)
                state_prime, reward, done = self.env.step(action_decoded)
                score += reward
                if reward>0:
                    buffer.add(state, action, state_prime, reward, probability[action])                             #John: 构造高效的batch化，不仅仅是在线选择
                state = state_prime
                #print('   ', 'learn:', '{0:>1}  ->  {1:>1}  :  {2:>2}'.format(state, action_decoded, reward))
                if done: break
            #print('learn:','score={:.2f}'.format(score),'best={:.2f}'.format(self.env.perfect),'\n')
            #
            losses = []
            R = 0
            for r, p in buffer.get_reward_probability(self.env)[::-1]:   #-1 means backward                       #John: 需要做一定策略的采样；单个就学习可能只学会单个    
                R = r + reward_decay * R                                                                          #John: 如果不是序列决策，做reward_decay就没有意义；    reward需要做归一化
                loss = -torch.log(p) * R                                                                          #John: 构建正确的学习支点
                losses.append(loss)
            if len(losses)>0:
                #print('learn', 'optimize network')
                losses = torch.stack(losses).sum()
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
            #
            if score >= self.env.perfect:
                print('learn:', 'try infer', '\n')
                if self.infer(show_info=1) == self.env.perfect: 
                    print('\n')
                    self.infer(show_info=1)
                    print('AGI $$$')
                    break

    def infer(self, total_infer=4, show_info=1):
        memory = Memory(size=self.env.width)
        state = self.env.free()
        for i in range(self.env.width):
            action = self.env.actions[i%len(self.env.actions)]
            state_prime, reward, done = self.env.step(action)
            memory.add_data_text(self.coding.encode(state), state)
            state = state_prime
        #
        scores = []
        state = self.env.free()
        for t in range(total_infer):
            score = 0
            done = False
            while not done:    
                memory.add_data_text(self.coding.encode(state), state) 
                state_memory = memory.get_last_data()
                state_memory = torch.tensor(state_memory, dtype=torch.long)
                state_memory = state_memory.view(-1, state_memory.shape[-1])
                probability = self.network(state_memory)[0]
                action = torch.argmax(probability.detach()).tolist() 
                action_decoded = self.coding.decode(action)
                state_prime, reward, done = self.env.step(action_decoded)
                score += reward
                state = state_prime
                if show_info: print('   ','infer:','{0:>1}  ->  {1:>1}  :  {2:>2}'.format(state, action_decoded, reward))
            scores.append(score)
            if show_info: print('infer:','score={:.2f}'.format(score),'best={:.2f}'.format(self.env.perfect),'\n')
        return sum(scores)/len(scores)

if __name__ == '__main__':
    Mathink(env=Math(show_sample=1)).learn()
