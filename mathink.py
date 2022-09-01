class TpjBrain:
    def __init__(self):
        self.index = -1
        self.hardcode = [TpjWorld.action_nop]*5 +[TpjWorld.actions[1],TpjWorld.actions[3],TpjWorld.actions[4]] + [TpjWorld.action_end] + [TpjWorld.action_nop] #65+69=134;

    def take(self, phenom):
        self.index += 1
        return self.hardcode[self.index]

class TpjWorld:  #RL-interactive-environment-style，Sequential，Multi-line: Addition, (TODO: Subtraction, Multiplication, Division; Any mathematic computing ...)   
    action_nop = ' '
    action_end = ';'
    actions = ['0','1','2','3','4','5','6','7','8','9']+[action_nop]+[action_end]

    def __init__(self, is_train, number_digit=2, random_seed=122333):
        self.state_add   = '+'
        self.state_equal = '='
        self.states = self.actions+[self.state_add]+[self.state_equal]
        self.reward_ok_yes = +0.9
        self.reward_ok_nop = +0.2
        self.reward_ok_end = +0.1
        self.reward_no_yes = -0.8
        self.reward_no_nop = -0.2
        self.reward_no_end = -0.1
        self.reward_do_any =  0.0
        self.perfect = self.reward_ok_nop*(number_digit+1+number_digit+1) + self.reward_ok_yes + self.reward_ok_end
        self.number_digit = number_digit
        self.dataset = []  
        number_total = (10**number_digit)**2
        number_train = int(number_total*0.8)
        generator = torch.Generator()
        generator.manual_seed(random_seed)  #same seed for reuse reproduce
        permutation = torch.randperm(number_total, generator=generator)
        indexes = permutation[:number_train] if is_train else permutation[number_train:]
        for index in indexes:
            index = index.item()
            nd = 10**number_digit
            a1 = index // nd
            a2 = index %  nd
            sm = a1 + a2
            a1str = f'%0{number_digit+0}d' % a1
            a2str = f'%0{number_digit+0}d' % a2
            smstr = f'%0{number_digit+1}d' % sm  #+1 for carry-overflow
            line = a1str + self.state_add + a2str + self.state_equal + smstr + self.action_end
            self.dataset.append(line)
        self.init()

    def init(self, init_line_random=True):
        self.offsetLine = torch.randint(0, len(self.dataset), (1,)).item() if init_line_random else 0
        self.offsetChar = -1
        self.response = ''
        return self.action_nop
    
    def take(self, action):
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
                    if self.offsetChar < len(self.dataset[self.offsetLine])-2:      #answer: ending-not
                        return self.reward_do_any
                    elif self.offsetChar < len(self.dataset[self.offsetLine])-1:    #answer: ending-yes ;
                        return self.reward_no_yes
                    else:                                                           #answer: long number
                        if action==self.action_end:                                 #  long number, ending-yes
                            return self.reward_ok_end
                        else:                                                       #  long number, ending-not
                            return self.reward_no_end
            else:  #question
                if action==self.action_nop:
                    return self.reward_ok_nop
                else:
                    return self.reward_no_nop

        _action(action) 
        state = _state()
        return state, _reward(), 1 if state==self.action_end else 0

class TPJ:
    def __init__(self):
        def set_seed(seed=122333):
            import random
            random.seed(seed)
            import numpy
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        set_seed()
        self.world = TpjWorld(is_train=True)
        self.brain = TpjBrain()

    def live(self):
        action = self.world.action_nop
        phenom = self.world.take(action)
        for i in range(1):
            for j in range(10):            
                action = self.brain.take(phenom)
                print('TPJ', 'phenom:',phenom, '->', 'action:', action)
                phenom = self.world.take(action)
            print()

if __name__ == '__main__':
    TPJ().live()
    
    
