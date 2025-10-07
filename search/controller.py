import math

from utils.config import ControllerArgs
import torch
import torch.nn.functional as F


class Controller(torch.nn.Module):

    def __init__(self, args: ControllerArgs):
        torch.nn.Module.__init__(self)
        self.args = args
        self.baseline = None

        self.num_tokens = []
        for idx in range(self.args.n_encoder - 1):
            self.num_tokens += [len(args.shared_rnn_activations), idx + 1]
        for idx in range(self.args.n_decoder - 1):
            self.num_tokens += [len(args.shared_rnn_activations), idx + 1]
        self.func_names = args.shared_rnn_activations
        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self.decoders = torch.nn.ModuleList(self.decoders)
        self.optim = torch.optim.Adam(self.parameters(), lr=args.lr)

        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs
        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_t
        if self.training is True:
            logits = (self.args.tanh_c * F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, detail=True):
        inputs = torch.zeros((1, self.args.controller_hid), device=next(self.parameters()).device, requires_grad=False)
        hidden = self.init_hidden(1)

        activations = []
        prev_nodes = []
        entropies = []
        log_probs = []
        for block_idx in range(2 * (self.args.num_blocks - 2)):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, action)
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = (action[:, 0] + sum(self.num_tokens[:mode])).data

            action = action[:, 0].tolist()
            if mode == 0:
                activations.append(action[0])
            elif mode == 1:
                prev_nodes.append(action[0])
        if detail:
            return activations, prev_nodes, torch.cat(log_probs), torch.cat(entropies)
        else:
            return activations, prev_nodes

    def step(self, rewards, log_probs):
        rewards = torch.flip(rewards, dims=(0,))
        for i in range(1, len(rewards)):
            rewards[i] = self.args.discount * rewards[i - 1] + rewards[i]
        rewards = torch.flip(rewards, dims=(0,))
        if self.baseline is None:
            self.baseline = rewards
        else:
            decay = self.args.baseline_decay
            self.baseline = decay * self.baseline + (1 - decay) * rewards
        adv = rewards - self.baseline
        loss = -log_probs * adv.clone().detach()
        loss = loss.sum()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.args.controller_hid),
                weight.new_zeros(bsz, self.args.controller_hid))

    def reset_history(self):
        self.baseline = None


if __name__ == "__main__":
    from utils.config import ctl_arg


    def get_reward(loss, entropies):
        ppl = math.exp(loss)
        reward = 80 / ppl
        rewards = reward + 0.0001 * entropies
        return rewards


    model = Controller(ctl_arg)
    activations, prev_nodes, log_probs, entropies = model.sample()
    print(activations, prev_nodes)
    rewards = get_reward(6, entropies)
    print(rewards, log_probs, sep='\n')
    model.step(rewards, log_probs)
