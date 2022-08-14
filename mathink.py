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

import numpy as np

class Coding:
    def __init__(self, i2t):
        self.unit_size = len(i2t)
        self.i2t = i2t
        self.t2i = {}
        i = 0
        for t in i2t:
            self.t2i[t] = i
            i += 1
    
    def encode(self, t):
        return self.t2i[t]
    
    def decode(self, i):
        return self.i2t[i]

class Memory:
    def __init__(self, coding, size=6):
        self.coding = coding
        self.size = size
        self.data = []
        for _ in range(self.size):
            self.data.append(Trans.onehot(self.coding.unit_size, self.coding.encode(Creator.action_nop)))    

    def add(self, x):
        if len(self.data) >= self.size:
            self.data = self.data[1:]
        self.data.append(Trans.onehot(self.coding.unit_size, self.coding.encode(Creator.action_nop)))

    def all(self):
        return self.data       

class Trans:
    def onehot(n, x):
        data = np.zeros((n,))
        data[x] = 1.0
        return data.tolist()   

    def squeeze(X):
        return np.tanh(X).tolist()

    def softmax(X):
        return (np.exp(X)/sum(np.exp(X))).tolist()

    def argmax(x):
        return np.argmax(x).tolist()
         
class Block:
    initialization = ['uniform','normal']
    
    def __init__(self, size, init=initialization[0]):
        self.W = np.random.uniform(low=0.0, high=1.0, size=size) if init==Block.initialization[0] else np.random.normal(loc=0.0, scale=1.0, size=size)
        self.b = np.zeros(shape=(1,))

    def perceive(self, X):
        return (np.dot(X, self.W) + self.b).tolist()[0]

    def decay(self):
        pass
    
    def motivate(self):
        pass
        
    def __repr__(self):
       return '\nW='+str(self.W) +'\n'+'b='+str(self.b)  
    
class Group:
    def __init__(self, block_size, block_number=3):
        self.block_all = [Block(size=(block_size,)) for _ in range(block_number)]   
        
    def perceive(self, X):  #route+attention
        linear_all = [block.perceive(X) for block in self.block_all]
        argmax = Trans.argmax(linear_all)
        softmax = Trans.softmax(linear_all)   #John: 通不通，是自己内部决定，和周边影响，而不是全局比较出来的
        actived_squeezed = Trans.squeeze(softmax)
        actived_weights = actived_squeezed[argmax]
        perceived = (np.array(X) * actived_weights).tolist()
        return perceived        

    def learn(self):
        pass    

class Brain:
    def __init__(self, coding):  #Todo: 构造功能区网络：[升维/多路冗余]，调度，记忆，计算，生成，(奖励利用)学习
        self.coding = coding
        self.groupDispatch = Group(block_size=coding.unit_size)
        self.DEBUG_i = -1      
                
    def think(self, state):  
        perceived = self.groupDispatch.perceive(state)

        self.DEBUG_i += 1
        if self.DEBUG_i == 5:
            DEBUG_onehot_action = Trans.onehot(self.coding.unit_size, self.coding.encode(Creator.actions[1]))
        elif self.DEBUG_i == 6:
            DEBUG_onehot_action = Trans.onehot(self.coding.unit_size, self.coding.encode(Creator.actions[0]))
        else:
            DEBUG_onehot_action = Trans.onehot(self.coding.unit_size, self.coding.encode(Creator.action_nop))       
        return DEBUG_onehot_action 
  
class Thinker:
    def __init__(self, creator):
        self.creator = creator
        self.coding = Coding(Creator.actions)
        self.memory = Memory(self.coding)
        self.brain = Brain(self.coding)
        
    def grow(self, epochs=3):       
        epoch = 0  
        state_old = self.creator.reset()
        while epoch < epochs: 
            action = self._think(state_old)          
            state_new, reward, done = self.creator.step(action)
            self.memory.add(state_old)
            if state_old!='': 
                print("{0:>2} -> {1:>2} : {2:>2}".format(state_old, action, reward))
            if state_old==Creator.action_separator: 
                epoch = epoch + 1
                print()   
            state_old = state_new 
            
    def _think(self, state):
        state_encoded = self.coding.encode(state)
        state_encoded_onehot = Trans.onehot(len(self.coding.i2t), state_encoded)
        action_thought = self.brain.think(state_encoded_onehot)
        action_thought_argmax = Trans.argmax(action_thought)
        action_thought_argmax_decoded = self.coding.decode(action_thought_argmax)
        return action_thought_argmax_decoded
   
import random
class Creator:  #强化环境，序列，多行，加法
    action_nop = ''
    action_separator = ';'
    action_equal = '='
    actions = ['0','1','2','3','4','5','6','7','8','9']+['+']+[action_equal]+[action_separator]+[action_nop]
    
    def __init__(self, range_max=50, sample_step=3):
        self.dataset = []  
        i = 0
        while i < range_max:
            j = 0
            while j < range_max:
                sample = str(i)+'+'+str(j)+'='+str(i+j)+Creator.action_separator 
                self.dataset.append(sample)          
                j = j + random.randint(1, sample_step)  #[1,3)
            i = i + random.randint(1, sample_step) 
        self.reset()

    def reset(self):
        self.offsetLine = 0 
        self.offsetChar = -1
        self.response = ''
        return self.__class__.action_nop
    
    def step(self, action):
        def _action():
            if action==self.__class__.action_nop:
                self.response = ''
            else:
                self.response += action
            
        def __next():
            self.offsetLine += 1 
            if self.offsetLine == len(self.dataset):
                self.offsetLine = 0
                
        def _state():
            if self.offsetLine == len(self.dataset):
                self.offsetLine = 0
            if self.offsetChar == len(self.dataset[self.offsetLine])-1:
                self.offsetChar = -1
                __next()             
            self.offsetChar += 1
            state = self.dataset[self.offsetLine][self.offsetChar]
            return state
            
        def _reward():
            equalIndex = self.dataset[self.offsetLine].index(Creator.action_equal)
            if self.offsetChar > equalIndex:
                answer = self.dataset[self.offsetLine][equalIndex+1:-1]
                if self.response == answer:
                    return +9
                else:
                    if self.offsetChar < len(self.dataset[self.offsetLine])-2:      #多位数答案中途，不惩，不奖
                        return 0
                    elif self.offsetChar < len(self.dataset[self.offsetLine])-1:    #多位数答案结束，惩罚
                        return -8
                    else:                                                           #行结束符，惩罚
                        if action==self.__class__.action_nop:
                            return -2
                        else:
                            return -3
            else:
                if action==self.__class__.action_nop:
                    return +1
                else:
                    return -1

        _action() 
        state = _state()
        return state, _reward(), state==Creator.action_separator
        
def main():
    creator = Creator()
    thinker = Thinker(creator)
    thinker.grow()

if __name__ == '__main__':
    main()
