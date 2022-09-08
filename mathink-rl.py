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
class Math:  #objective-environment#math-problem/客观世界/数学问题：muliti-line多行，addition/加法, interactive/交互式, reward-as-feedback/奖励作为反馈; TODO: include all math knowledge in NLP
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
 
        self.dataset = []    #so-far, not split train and valid
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
            print('Mathink: dataset for learn and infer:')
            for one in self.dataset:
                print(' ',one)
            print()

    def free(self, init_line_random=True):
        self.offsetLine = random.randint(0, len(self.dataset)) if init_line_random else 0
        self.offsetChar = -1
        self.response = ''
        return self.action_nop
    
    def step(self, action):
        def _action(action):
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

        _action(action) 
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

import torch
class Encoder(torch.nn.Module):
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
            m = e + p
            return self.dropout(m)

    def __init__(self, intput_length, input_dimension, hidden_dimension=32):
        super().__init__()
        self.embeder = torch.nn.Embedding(intput_length*input_dimension, hidden_dimension)
        self.encoder = self.__class__.PositionalEncoding(intput_length, hidden_dimension)

    def forward(self, x):
        o = self.embeder(x)
        o = self.encoder(o)
        return o

class Transformer(torch.nn.Module):
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
            import math
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

    def __init__(self, intput_length, output_dimensionm, hidden_dimension=32, backbone_size=2):
        super().__init__()
        self.backbone = torch.nn.ModuleList()
        for _ in range(backbone_size):
            self.backbone.append(self.__class__.MultiHeadSelfAttention(hidden_dimension, hidden_dimension, hidden_dimension))
        self.gather = torch.nn.Sequential(torch.nn.Linear(intput_length*hidden_dimension, hidden_dimension), torch.nn.ReLU(), torch.nn.Linear(hidden_dimension, output_dimensionm), torch.nn.Softmax(dim=-1))  #softmax for Categorical
   
    def forward(self, x):
        o = x
        for layer in self.backbone:
            o = layer(o)
        o = o.reshape(o.shape[0], o.shape[1]*o.shape[2])
        o = self.gather(o) #.squeeze()
        return o

class MoE(torch.nn.Module):  #sparsely gated mixture of experts, original paper use noise for load balancing
    class SparseDispatcherCombiner:
        def __init__(self, num_experts, gates):
            self._num_experts = num_experts
            self._gates = gates  #[batch_size, num_experts]
            sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  #非0的gate排序（值和索引）       
            _, self._expert_index = sorted_experts.split(1, dim=1)   # drop indices        
            self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]  #通过batch的索引来控制给每专家的样本
            self._part_sizes = list((gates > 0).sum(0).numpy())  #每个专家的样本数        
            gates_exp = gates[self._batch_index.flatten()]  # expand gates to match with self._batch_index
            self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  #gather收集

        def dispatch(self, inp):  #[batch_size, input_size]  #分派样本给gate非0的专家
            inp_exp = inp[self._batch_index].squeeze(1)  # expand according to batch index so we can just split by _part_sizes
            dispatched = torch.split(inp_exp, self._part_sizes, dim=0)  #list: [expert_batch_size_i, <extra_input_dims>]
            return dispatched

        def combine(self, expert_out, multiply_by_gates=True):  #组合专家输出成统一的输出张量
            stitched = torch.cat(expert_out, 0).exp()  #map to exp space
            if multiply_by_gates:
                stitched = stitched.mul(self._nonzero_gates)
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True)
            combined = zeros.index_add(0, self._batch_index, stitched.float()) #combine samples processed by same expert
            combined[combined == 0] = 2.220446049250313e-16  #if 0, add eps, avoid NaN when back to log space   #numpy.finfo(float).eps
            return combined.log()

    def __init__(self, input_size, experts, top_k):
        super().__init__()
        self.input_size = input_size
        self.num_experts = len(experts)
        self.top_k = top_k
        assert(self.top_k <= self.num_experts)
        self.experts = torch.nn.ModuleList(experts)
        self.w_gate = torch.nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.softmax = torch.nn.Softmax(1)

    def _top_k_gating(self, x):  #x*w -> top_k -> softmax  -> scatter/分散
        logits = x @ self.w_gate  #@ is dot in python
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=1)  #top-k mask for sparse
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = (gates > 0).sum(0)  #true load per expert, load is number of examples for which the corresponding gate>0 
        return gates, load  #[batch_size, num_experts], [num_experts]

    def _variation_coefficient_squared(self, x, eps=1e-10):  #squared coefficient of variation of a sample, encourage positive distribution more uniform
        if x.shape[0] == 1:  #only num_experts = 1
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, loss_coef):
        flat = x.view(-1, x.shape[1]*x.shape[2])
        gates, load = self._top_k_gating(flat)
        importance = gates.sum(0)  #importance loss  #experts的gates的方差系数平方，作为损失，来让方差变小而exports比较平衡
        balancing_loss = self._variation_coefficient_squared(importance) + self._variation_coefficient_squared(load)  #encourages all experts to be approximately equally used across a batch
        balancing_loss *= loss_coef  #multiplier on load-balancing losses
        dispatcher = self.__class__.SparseDispatcherCombiner(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        #print('expert_inputs', [one.shape[0] for one in expert_inputs])
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, balancing_loss

class Network(torch.nn.Module):
    def __init__(self, intput_length, input_dimension, output_dimensionm, hidden_dimension=32, num_experts=1, top_k=1):
        super().__init__()
        self.moe = 0
        self.encoder = Encoder(intput_length=intput_length, input_dimension=input_dimension, hidden_dimension=hidden_dimension)
        if self.moe:
            experts = [Transformer(intput_length=intput_length, output_dimensionm=output_dimensionm, hidden_dimension=hidden_dimension) for i in range(num_experts)] 
            self.backbone = MoE(intput_length*hidden_dimension, experts, top_k)
        else:
            self.backbone = Transformer(intput_length=intput_length, output_dimensionm=output_dimensionm, hidden_dimension=hidden_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, loss_coef=0.01):
        e = self.encoder(x)
        if self.moe:
            o, balancing_loss = self.backbone(e, loss_coef)   #balancing_loss in batch, not cross-batch
        else:
            o, balancing_loss = self.backbone(e), 0
        o = self.softmax(o)
        return o[0], balancing_loss   #online, only-one

class Mathink:
    def __init__(self, env):
        self.env = env       
        self.network = Network(self.env.width, len(self.env.states), len(self.env.actions))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.coding = Coding(self.env.states)

    def learn(self, total_learn=30000, reward_decay=0.98):
        class ReinforceBuffer:  #algorithm-specific
            def __init__(self, size):
                self.size = size
                self.data = []

            def add(self, state, action, state_prime, reward, probability_action, balancing_loss):
                if len(self.data) >= self.size:
                    self.data = self.data[1:]
                self.data.append((reward, probability_action, balancing_loss))

            def get_reward_probability(self, env):
                r9 = 0
                r2 = 0
                r1 = 0
                for r,p, balance_loss in self.data:
                    if r==env.reward_ok_yes:
                        r9 = 1
                    elif r==env.reward_ok_nop:
                        r2 = 1  
                    elif r==env.reward_ok_end:
                        r1 = 1
                    else:
                        pass  
                return self.data if ((r9 and r2) or (r9 and r1) or (r2 and r1)) else []
        memory = Memory(size=self.env.width)
        state = self.env.free()
        for i in range(self.env.width):  #warm-up
            action = self.env.actions[i%len(self.env.actions)]
            state_prime, reward, done = self.env.step(action)
            memory.add_data_text(self.coding.encode(state), state)
            state = state_prime
        #
        state = self.env.free()
        for t in range(total_learn):
            buffer = ReinforceBuffer(size=64)
            score = 0
            done = False
            while not done:   
                memory.add_data_text(self.coding.encode(state), state)                     #John: 构造好的记忆，要么结构化，要么网络本身
                state_memory = memory.get_last_data()
                state_memory = torch.tensor(state_memory, dtype=torch.long)                #John: 必须包含必要的信息比如各种结构信息，并有合适的网络去抓取
                state_memory = state_memory.view(-1, state_memory.shape[-1])
                probability, balancing_loss = self.network(state_memory)                   #John: 在线逐样本学习效率不高
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
                if reward > 0:
                    buffer.add(state, action, state_prime, reward, probability[action], balancing_loss)           #John: 构造高效的batch化，不仅仅是在线选择
                state = state_prime
                #print('   ', 'learn:', '{0:>1}  ->  {1:>1}  :  {2:>2}'.format(state, action_decoded, reward))
                if done: break
            if t%100==0: print('learn:','{:05d}'.format(t),'current-learned-score={:+.2f}'.format(score),'expected-best-score={:+.2f}'.format(self.env.perfect), ' ', 'XX' if score<self.env.perfect else 'OK')
            #
            losses = []
            R = 0
            for r, p, balance_loss in buffer.get_reward_probability(self.env)[::-1]:   #-1 means backward         #John: 需要做一定策略的采样；单个就学习可能只学会单个    
                R = r + reward_decay * R                                                                          #John: 如果不是序列决策，做reward_decay就没有意义；    reward需要做归一化
                reward_loss = -torch.log(p) * R                                                                   #John: 构建正确的学习支点
                #print('reward_loss=', reward_loss, '  ', 'balance_loss=', balance_loss)
                loss = reward_loss + balance_loss
                losses.append(loss)
            if len(losses)>0:
                #print('learn', 'optimize network')
                losses = torch.stack(losses).sum()
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
            #
            if score >= self.env.perfect:
                print('learn:','score:{:+.2f}'.format(score), 'perfect:{:+.2f}'.format(self.env.perfect))
                average_score = self.infer(show_info=1)
                if average_score == self.env.perfect: 
                    self.infer(show_info=1)
                    print('TPJ SuperAI Mathink Perfect $$$')
                    break

    def infer(self, total_infer=4, show_info=0):
        memory = Memory(size=self.env.width)
        state = self.env.free()
        for i in range(self.env.width):
            action = self.env.actions[i%len(self.env.actions)]
            state_prime, reward, done = self.env.step(action)
            memory.add_data_text(self.coding.encode(state), state)
            state = state_prime
        scores = []
        state = self.env.free()
        self.network.eval()
        for t in range(total_infer):
            score = 0
            done = False
            while not done:    
                memory.add_data_text(self.coding.encode(state), state) 
                state_memory = memory.get_last_data()
                state_memory = torch.tensor(state_memory, dtype=torch.long)
                state_memory = state_memory.view(-1, state_memory.shape[-1])
                probability, balancing_loss = self.network(state_memory)
                action = torch.argmax(probability.detach()).tolist() 
                action_decoded = self.coding.decode(action)
                state_prime, reward, done = self.env.step(action_decoded)
                score += reward
                state = state_prime
                if 0: print('   ','infer:','{0:>1}  ->  {1:>1}  :  {2:>2}'.format(state, action_decoded, reward))
            scores.append(score)
            if show_info: print('   ','infer:','current-learned-score={:+.2f}'.format(score),'expected-best-score={:+.2f}'.format(self.env.perfect), ' ', 'XX' if score<self.env.perfect else 'OK')
        self.network.train()
        average_score = sum(scores)/len(scores)
        print('   ','infer:','current-average-score={:+.2f}'.format(average_score),'>>>')
        return average_score

if __name__ == '__main__':
    def set_seed(seed=None):
        import random, numpy
        if seed is None: 
            seed = random.randint(0,2**32)
            print('random-seed:', seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed()
    Mathink(env=Math(show_sample=1)).learn()
