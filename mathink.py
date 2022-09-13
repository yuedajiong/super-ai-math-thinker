'''

########  Sutton
#all four components are updated by learning processes operating in the foreground using the most recent events together with short-term credit-assignment memories such as eligibility traces

class Perception:
    def __init__(self):
        pass
    #summary of its past experience, state  #foreground
    #input: observation, last-action, state
    #output: state

class ReactivePolices:
    def __init__(self):
        pass
    #goal of maximizing reward  #foreground
    #input: state, value
    #output: action
    #
    #primary policy     #multiple policies and value functions 
    #xxxx policy: 

class ValueFunctions:   #multiple policies and value functions 
    def __init__(self):
        pass
    #corresponding to each policy  #foreground
    #input: reward, state
    #output: value
    
class TransitionModel:
    def __init__(self):
        pass
    #knowledge of the world’s dynamics  #temporally abstract(action or option), planning  #asynchronously, background
    #input: observed-actions, rewards, states;  without-observations
    #output: next state, next reward

#########  LeCun

class Configurator:
    def __init__(self):
        pass
    #->  Perception  Actor  Critic  WorldModel

class Perception:
    def __init__(self):
        pass
    #->  Actor  Critic  WorldModel
    #<-  ObjectiveWorld

class ShortTermMemory:
    def __init__(self):
        pass
    #<>  WorldModel

    class KeyValueMemoryNetworks:
        def __init__(self):
            pass

    class QueueMemoryNetworks:
        def __init__(self):
            pass

    class GraphMemoryNetworks:
        def __init__(self):
            pass

class Actor:
    def __init__(self):
        pass
    #->  Configurator  Perception
    #<-  WorldModel  ObjectiveWorld

class Critic:  #train-data-from-
    def __init__(self):
        pass
    #<-  Configurator  Perception  WorldModel  

    class IntrinsicCost:
        def __init__(self):
            pass

    class TrainableCost:
        def __init__(self):
            loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    class ShortTermAssociativeMemoryNetwork:
        def __init__(self):
            pass

class WorldModel:
    def __init__(self):
        pass
    #->  Critic
    #<-  Configurator  Perception
    #<>  ShortTermMemory

###

class Hierarchical:
    def __init__(self):
        pass
    #lower I/O encoded to upper I/O encoded

class Planning:
    def __init__(self):
        pass
    #lower predicted to upper actor

###

class Embeder:    #原始输入层，可选，符号类离散输入
    def __init__(self, vocab_size, block_size, n_embd=48, embd_pdrop=0.1):
        self.wte=torch.nn.Embedding(vocab_size, n_embd)
        self.wpe=torch.nn.Embedding(block_size, n_embd)
        self.drop=torch.nn.Dropout(embd_pdrop)

    def forward(self, idx, targets=None):  #Batch
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)  #shape (1, t)
        tok_emb = self.transformer.wte(idx)  #token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  #position embeddings of shape (1, t, n_embd)
        e = self.transformer.drop(tok_emb + pos_emb)
        return e

class Encoder:
    class Backbone(torch.nn.Module):
        class Block(torch.nn.Module):
            class CausalSelfAttention(torch.nn.Module):  #multi-head masked self-attention -> projection
                def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1):
                    super().__init__()
                    self.n_embd = n_embd
                    self.n_head = n_head
                    self.c_attn = torch.nn.Linear(self.n_embd, self.n_embd * 3)
                    self.attn_dropout = torch.nn.Dropout(attn_pdrop)
                    self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd)
                    self.resid_dropout = torch.nn.Dropout(resid_pdrop)
                    self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))  #causal mask to ensure that attention is only applied to the left in the input sequence

                def forward(self, x):
                    B, T, C = x.size()  #batch-size, sequence-length, embedding-dimensionality (n_embd)
                    q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
                    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                    att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1))**0.5)  #causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                    att = torch.torch.nn.functional.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v  #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                    y = y.transpose(1, 2).contiguous().view(B, T, C)  #re-assemble all head outputs side by side
                    y = self.resid_dropout(self.c_proj(y))  #output projection
                    return y

            class GELU(torch.nn.Module):  #Gaussian Error Linear Units (GELU) https://arxiv.org/abs/1606.08415
                def forward(self, x):
                    return 0.5 * x * (1.0 + torch.tanh((2.0/torch.pi)**0.5 * (x + 0.044715 * torch.pow(x, 3.0))))

            def __init__(self, n_embd, n_head, block_size, resid_pdrop=0.1):
                super().__init__()
                self.ln_1 = torch.nn.LayerNorm(n_embd)
                self.attn = self.__class__.CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop=resid_pdrop)
                self.ln_2 = torch.nn.LayerNorm(n_embd)
                self.mlp = torch.nn.ModuleDict(dict(c_fc=torch.nn.Linear(n_embd, 4 * n_embd), c_proj=torch.nn.Linear(4 * n_embd, n_embd), act=self.__class__.GELU(), dropout=torch.nn.Dropout(resid_pdrop)))
                self.mlpf = lambda x: self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))))

            def forward(self, x):
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlpf(self.ln_2(x))
                return x

    def __init__(self, block_size, n_embd=48, n_layer=3, n_head=3):  #maximize information content:   Exp-and<variance&covariance -> Identity >;    John:reconstruction 
        self.blocks = torch.nn.ModuleList([self.__class__.Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        for block in self.block:
            x = block(x)
        x = self.ln_f(x)
        return x

    def Projector(args, embedding):  #胶合Glue
        mlp_spec = f"{embedding}-{args.mlp}"  #resnet50:2048  8192-8192-8192
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):  #[2048,8192]
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))  #[8192,8192]
        return nn.Sequential(*layers)

class Predictor:  #具体任务用
    def __init__(self, vocab_size, n_embd):
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):  #input is encoded-x and z
        o = self.lm_head(x)
        return o

class Z:
    def __init__(self):
        pass
    #minimize information content:  quantization/VQ-VAE, min-rank, sparse, denoise 
    #sample / uplayer / ...
    #vector

class VICRegularizationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.batch_size = 2     #可以动态获取
        self.num_features = 64  #可以动态获取

    def forward(self, x, y):
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        repr_loss = torch.nn.functional.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

def compute_l1_loss(self, z):
  return torch.abs(z).sum()

def compute_l2_loss(self, z):
  return torch.square(z).sum()

#torch.nn.functional.one_hot(target,num_classes=100)
#torch.argmax(one_hot, dim=1)

class TpjBrain(torch.nn.Module):
    class Block(torch.nn.Module):
        class CausalSelfAttention(torch.nn.Module):  #multi-head masked self-attention -> projection
            def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1):
                super().__init__()
                self.n_embd = n_embd
                self.n_head = n_head
                self.c_attn = torch.nn.Linear(self.n_embd, self.n_embd * 3)
                self.attn_dropout = torch.nn.Dropout(attn_pdrop)
                self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd)
                self.resid_dropout = torch.nn.Dropout(resid_pdrop)
                self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))  #causal mask to ensure that attention is only applied to the left in the input sequence

            def forward(self, x):
                B, T, C = x.size()  #batch-size, sequence-length, embedding-dimensionality (n_embd)
                q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #(B, nh, T, hs)
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1))**0.5)  #causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = torch.torch.nn.functional.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y = y.transpose(1, 2).contiguous().view(B, T, C)  #re-assemble all head outputs side by side
                y = self.resid_dropout(self.c_proj(y))  #output projection
                return y

        class GELU(torch.nn.Module):  #Gaussian Error Linear Units (GELU) https://arxiv.org/abs/1606.08415
            def forward(self, x):
                return 0.5 * x * (1.0 + torch.tanh((2.0/torch.pi)**0.5 * (x + 0.044715 * torch.pow(x, 3.0))))

        def __init__(self, n_embd, n_head, block_size, resid_pdrop=0.1):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(n_embd)
            self.attn = self.__class__.CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop=resid_pdrop)
            self.ln_2 = torch.nn.LayerNorm(n_embd)
            self.mlp = torch.nn.ModuleDict(dict(c_fc=torch.nn.Linear(n_embd, 4 * n_embd), c_proj=torch.nn.Linear(4 * n_embd, n_embd), act=self.__class__.GELU(), dropout=torch.nn.Dropout(resid_pdrop)))
            self.mlpf = lambda x: self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))))

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
            return x

    def __init__(self, vocab_size, block_size, model_type='gpt-nano', n_embd=48, embd_pdrop=0.1, n_layer=3, n_head=3, ):
        super().__init__()
        self.block_size = block_size
        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(vocab_size, n_embd),
            wpe=torch.nn.Embedding(block_size, n_embd),
            drop=torch.nn.Dropout(embd_pdrop),
            h=torch.nn.ModuleList([self.__class__.Block(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f=torch.nn.LayerNorm(n_embd)))
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)
        def init_weights(module):
            if isinstance(module, torch.nn.Linear):
                torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.torch.nn.init.zeros_(module.bias)
                torch.torch.nn.init.ones_(module.weight)
        self.apply(init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):  #a special scaled init to the residual projections
                torch.torch.nn.init.normal_(p, mean=0.0, std=0.02/((2 * n_layer)**0.5))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)  #shape (1, t)
        tok_emb = self.transformer.wte(idx)  #token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  #position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def learn(self, idx, targets):
        logits = self.forward(idx)
        loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def infer(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):  #take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete the sequence max_new_tokens times, feeding the predictions back into the model each time.
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]  #if the sequence context is growing too long we must crop it at block_size
            logits = self.forward(idx_cond)

            logits = logits[:, -1, :] / temperature  #pluck the logits at the final step and scale by desired temperature
            if top_k is not None:  #optionally crop the logits to only the top k options
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.torch.nn.functional.softmax(logits, dim=-1)  #apply softmax to convert logits to (normalized) probabilities
            if do_sample:  #either sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:  #or take the most likely element
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)  #append sampled index to the running sequence and continue
        return idx

class TpjLearn:
    def __init__(self, brain, train_dataset, valid_dataset, weight_decay=0.1, betas=(0.9, 0.95)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = brain.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        ###
        no_decay = set()
        do_decay = set()
        for mn, m in brain.named_modules():
            for pn, p in m.named_parameters():   #named_modules is recursive, use set() to filter
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (torch.torch.nn.LayerNorm, torch.torch.nn.Embedding)):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (torch.torch.nn.Linear)):
                    do_decay.add(fpn)
        param_dict = {pn: p for pn, p in brain.named_parameters()}
        optim_groups = [{"params": [param_dict[pn] for pn in sorted(list(do_decay))], "weight_decay": weight_decay}, {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=5e-4, betas=betas)
        ###
        self.optimizer = torch.optim.AdamW(brain.parameters(), lr=5e-4, betas=betas)

    def valid(self, dataset):
        ndigit = self.train_dataset.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(self.device)
        loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=64, num_workers=1, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(self.device)
            d1d2 = x[:, :ndigit*2]  #isolate the first two digits of the input sequence alone
            d1d2d3 = self.brain.infer(d1d2, ndigit+1, do_sample=False) # using greedy argmax, not sampling
            d3 = d1d2d3[:, -(ndigit+1):]  #isolate the last digit of the sampled sequence
            d3 = d3.flip(1) #reverse the digits to their "normal" order
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)   # decode the integers from individual digits
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)  # decode the integers from individual digits
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i #manually calculate the ground truth
            correct = (d3i_pred == d3i_gt).cpu()
            for i in range(x.size(0)):
                results.append(int(correct[i]))
        rt = torch.tensor(results, dtype=torch.float)
        print("valid correct: %d/%d = %.2f%%"%(rt.sum(), len(results), 100*rt.mean()))
        return rt.mean()

    def learn(self, grad_norm_clip=1.0, max_iters=5000):
        train_loader = torch.utils.data.dataloader.DataLoader(self.train_dataset, batch_size=64, num_workers=1, drop_last=False, pin_memory=True, shuffle=False, sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)))
        self.brain.train()
        top_score = 0.0
        data_iter = iter(train_loader)
        for iter_num in range(max_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                raise
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            logits, self.loss = self.brain.learn(x, y)
            self.brain.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.torch.nn.utils.clip_grad_norm_(self.brain.parameters(), grad_norm_clip)
            self.optimizer.step()
            if iter_num % 500 == 0:
                train_max_batches = {1: None, 2: None, 3: 5}[self.train_dataset.ndigit] # if ndigit=2 we can afford the whole train set, ow no
                self.brain.eval()
                with torch.no_grad():
                    train_score = self.valid(self.train_dataset)
                    valid_score  = self.valid(self.valid_dataset)
                score = train_score + valid_score
                if score > top_score:
                    top_score = score
                    print("iter", iter_num, "save model, top score %.4f"%(score))
                    import os
                    ckpt_path = os.path.join('./chkp/', "model.pt")
                    if not os.path.exists(ckpt_path): os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(self.brain.state_dict(), ckpt_path)
                self.brain.train()
                if score==1.0*2:
                    break
                else:
                    print()

if __name__ == '__main__':
    train_dataset = TpjWorld(split='train')
    valid_dataset = TpjWorld(split='valid')
    brain = TpjBrain(train_dataset.get_vocab_size(), train_dataset.get_block_size())
    learn = TpjLearn(brain, train_dataset, valid_dataset)
    learn.learn()

if 1:
    hardcode = [TpjWorld.state_action_nop]*5 +[TpjWorld.state_action_all[1],TpjWorld.state_action_all[3],TpjWorld.state_action_all[4]] + [TpjWorld.state_action_end] + [TpjWorld.state_action_nop]  #65+69=134;  World:random_seed=122333@(generate&init)
    if not hasattr(self, 'index') or self.index >= len(hardcode)-1:
        self.index = 0
    else:
        self.index += 1
    return hardcode[self.index]

'''

import torch

class TpjWorld:  #RL-interactive-environment-style，Sequential，Multi-line: Addition, (TODO#1: Active-Mode; TODO#2: Subtraction, Multiplication, Division; Any mathematic ability ...) 
    state_add = '+'
    state_out = '='  
    state_action_end = ';'
    state_action_nop = ' '
    state_action_all = ['0','1','2','3','4','5','6','7','8','9']+[state_add]+[state_out]+[state_action_end]+[state_action_nop]

    def __init__(self, is_train, number_digit=2, reproduce_random_seed=122333):
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
        generator.manual_seed(reproduce_random_seed)  #same seed for reuse reproduce
        permutation = torch.randperm(number_total, generator=generator)
        indexes = permutation[:number_train] if is_train else permutation[number_train:]
        for index in indexes:
            index = index.item()
            nd = 10**number_digit
            a1 = index // nd
            a2 = index %  nd
            sm = a1 + a2
            a1_str = f'%0{number_digit+0}d' % a1
            a2_str = f'%0{number_digit+0}d' % a2
            sm_str = f'%0{number_digit+1}d' % sm  #+1 for carry-overflow
            line = a1_str + self.state_add + a2_str + self.state_out + sm_str + self.state_action_end
            self.dataset.append(line)
        self.task_length = len(self.dataset[0])
        self.init()

    def init(self, init_line_random=True):
        self.offsetLine = torch.randint(0, len(self.dataset), (1,)).item() if init_line_random else 0
        self.offsetChar = -1
        self.response = ''
        return self.state_action_nop
    
    def take(self, action):
        def _action(action):
            if action==self.state_action_nop:
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
            equalIndex = self.dataset[self.offsetLine].index(self.state_out)
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
                        if action==self.state_action_end:                           #  long number, ending-yes
                            return self.reward_ok_end
                        else:                                                       #  long number, ending-not
                            return self.reward_no_end
            else:  #question
                if action==self.state_action_nop:
                    return self.reward_ok_nop
                else:
                    return self.reward_no_nop

        _action(action) 
        state = _state()
        return state, _reward(), (1 if state==self.state_action_end else 0)

class TpjQueue:   #not neural so far
    def __init__(self, size):
        self.size = size
        self.data = []

    def add_data(self, data):
        if len(self.data) >= self.size:
            self.data = self.data[1:]
        self.data.append(data)

    def get_data(self):
        return self.data if len(self.data)==self.size else None

class TpjEncoder(torch.nn.Module):
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, max_length, hidden_dimension):
            super().__init__()
            self.P = torch.zeros((1, max_length, hidden_dimension))
            x = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, hidden_dimension, 2, dtype=torch.float32) / hidden_dimension)
            self.P[:, :, 0::2] = torch.sin(x)
            self.P[:, :, 1::2] = torch.cos(x)

        def forward(self, e):
            p = self.P[: , :e.shape[1] , :]
            return p

    def __init__(self, input_dimension, intput_length, encode_dimension=32, dropout_ratio=0.1):
        super().__init__()
        self.embeder = torch.nn.Embedding(input_dimension, encode_dimension)
        self.encoder = self.__class__.PositionalEncoding(intput_length, encode_dimension)
        self.dropout = torch.nn.Dropout(dropout_ratio)

    def forward(self, x):
        m = self.embeder(x)
        n = self.encoder(m)
        o = m + n
        return self.dropout(o)

class TpjBrain:  
    def __init__(self, task_length, task_dimension, memory_size=10):
        self.queue = TpjQueue(memory_size)
        self.encoder = TpjEncoder(input_dimension=task_dimension, intput_length=task_length)

    def think(self, state, reward, done):
        self.queue.add_data(state)
        state_queued = self.queue.get_data()
        if state_queued:
            state_queued = torch.tensor(state_queued, dtype=torch.long)                
            state_queued = state_queued.view(-1, state_queued.shape[-1])  #[1, 10]
            encoded = self.encoder(state_queued)
            print('encoded', encoded.shape)   #[1, 10, 32]  intput_length=10   encode_dimension=32

        action = 13  #TpjWorld.state_action_nop  ' '
        return action

class TpjAgent:
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

    def __init__(self, task_length, state_action_all):
        self.coding =  self.__class__.Coding(state_action_all)
        self.brain = TpjBrain(task_length, task_dimension=len(state_action_all))

    def take(self, phenom):
        state, reward, done = phenom[0], phenom[1], phenom[2]  #('4', 0.9, 0)
        encoded_state = self.coding.encode(state)
        action = self.brain.think(encoded_state, reward, done)
        decoded_action = self.coding.decode(action)
        return decoded_action

class Tpj:
    def __init__(self):
        self.world = TpjWorld(is_train=True)
        self.agent = TpjAgent(self.world.task_length, self.world.state_action_all)

    def live(self, life=2):
        action = self.world.state_action_nop
        phenom = self.world.take(action)
        for i in range(life):
            for j in range(10):  #task-specifical            
                action = self.agent.take(phenom)
                print('TPJ', 'phenom:',phenom, '->', 'action:', action)
                phenom = self.world.take(action)
            print()

if __name__ == '__main__':
    def set_random_seed(seed=122333):
        import random; random.seed(seed)
        import numpy; numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_random_seed()
    Tpj().live()

