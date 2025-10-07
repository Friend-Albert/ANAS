import json
import os
import random

import torch
import torch.nn as nn

from search.SelfAttention import ScaledDotProductAttention
from .dagcell import DagCell

from utils.config import RnnArgs


def create_file_if_not_exists(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, p=0.5):
        if not self.training or not p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - p)
        mask = mask.div_(1 - p)
        mask = mask.expand_as(x)
        return x * mask


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


def gen_mask(input, rate):
    mask = torch.zeros(input.shape, device=input.device, requires_grad=False).bernoulli_(1 - rate)
    mask = mask.div_(1 - rate)
    return mask


class DagModel(nn.Module):
    def __init__(self, encoder_args: RnnArgs, decoder_args: RnnArgs, batch_first=True, mode="shared"):
        super(DagModel, self).__init__()
        args = encoder_args
        self.args = encoder_args
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.mode = mode
        self.batch_first = batch_first
        self.rectify_factor = 1.0
        self.embedding = nn.Linear(self.args.n_input, self.args.n_emb)
        self.fc_out = nn.Linear(self.decoder_args.n_hidden, self.decoder_args.n_output)
        self.encoder = DagCell(encoder_args.n_node, encoder_args.n_emb, encoder_args.n_hidden)
        self.decoder = DagCell(decoder_args.n_node, encoder_args.n_emb + encoder_args.attention_output,
                            decoder_args.n_hidden)
        self.attention = ScaledDotProductAttention(d_model=args.n_hidden,
                                                   d_k=args.d_k,
                                                   d_v=args.d_v,
                                                   out_dim=args.attention_output,
                                                   h=args.attention_head)

        self.init_weights()
        self.resample()

    def init_weights(self):
        init_range = 0.025 if self.mode == "shared" else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.fc_out.bias.data.zero_()

    def resample(self, policy='rand', activation=None, prev_node=None):
        if 'rand' in policy:
            self.rand_sample()
        elif 'LSTM' in policy:
            self.construct(activation, prev_node)

    def forward(self, input, detail=False):
        if self.batch_first:
            input = input.transpose(0, 1)
        steps, bsz, n_input = input.shape
        hidden = self.init_hidden(bsz, self.encoder_args.n_hidden)
        emb = self.embedding(input)

        outputs = []
        for step in range(steps):
            raw_output, hidden = self.encoder(emb[step], hidden)
            outputs.append(raw_output)
            hidden = self.hidden_norm(hidden)

        h_decoder = outputs[-1]
        output = torch.stack(outputs, dim=1)
        output = self.attention(output, output, output)
        output = torch.cat([emb, output], dim=2)
        outputs.clear()

        for step in range(steps):
            raw_output, h_decoder = self.decoder(output[step], h_decoder)
            outputs.append(raw_output)
            h_decoder = self.hidden_norm(h_decoder)

        output = torch.stack(outputs)
        decoder_out = output
        result = self.fc_out(output)[-1, :, :]
        if detail:
            return result, h_decoder, decoder_out
        return result

    def hidden_norm(self, hidden):
        if self.mode == 'shared':
            h_norms = hidden.norm(dim=-1)
            max_norm = 25.0
            if h_norms.max() > max_norm:
                mask = torch.ones_like(hidden, requires_grad=False)
                mask[h_norms > max_norm] *= max_norm / h_norms[h_norms > max_norm][:, None]
                hidden = mask * hidden
        return hidden

    def init_hidden(self, bsz, n_hidden):
        weight = next(self.parameters())
        return weight.new_zeros(bsz, n_hidden)

    def load(self, best_path):
        with open(best_path, 'r') as json_file:
            info = json_file.readline()
            info = json.loads(info)
        self.encoder.construct(info['encoder_acti'], info['encoder_archi'])
        self.decoder.construct(info['decoder_acti'], info['decoder_archi'])

    def save(self, path, mode='w', info=None):
        create_file_if_not_exists(path)
        if info is None:
            info = {}
        info['encoder_acti'] = self.encoder.activation
        info['encoder_archi'] = self.encoder.architecture
        info['decoder_acti'] = self.decoder.activation
        info['decoder_archi'] = self.decoder.architecture
        with open(path, mode) as json_file:
            json.dump(info, json_file)
            json_file.write('\n')

    def get_architecture(self):
        activation = self.encoder.activation + self.decoder.activation
        architecture = self.encoder.architecture + self.decoder.architecture
        return activation, architecture

    def rand_sample(self):
        self.decoder.rand_sample()
        self.encoder.rand_sample()

    def perturb(self, activation, architecture):
        encoder_acti, decoder_acti = activation[:self.encoder_args.n_node - 1], activation[
                                                                                self.encoder_args.n_node - 1:]
        encoder_archi, decoder_archi = architecture[:self.encoder_args.n_node - 1], architecture[
                                                                                    self.encoder_args.n_node - 1:]
        ran = random.random()
        if ran <= 0.70:
            self.encoder.perturb(encoder_acti, encoder_archi, 2)
        elif 0.70 < ran <= 0.90:
            self.encoder.perturb(encoder_acti, encoder_archi, 1)
            self.decoder.perturb(decoder_acti, decoder_archi, 1)
        else:
            self.decoder.perturb(decoder_acti, decoder_archi, 2)
        self.rectify_factor = 0.96

    def construct(self, activation, architecture):
        encoder_acti, decoder_acti = activation[:self.args.n_node - 1], activation[self.args.n_node - 1:]
        encoder_archi, decoder_archi = architecture[:self.args.n_node - 1], architecture[self.args.n_node - 1:]
        self.encoder.construct(encoder_acti, encoder_archi)
        self.decoder.construct(decoder_acti, decoder_archi)


class WassersteinLoss(nn.Module):
    def __init__(self, num_bins):
        super(WassersteinLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        bsz = y_pred.shape[0]
        # 找到预测值和真实值的最大和最小值
        min_val = min(torch.min(y_pred).item(), torch.min(y_true).item())
        max_val = max(torch.max(y_pred).item(), torch.max(y_true).item())

        # 将预测值和真实值划分成n个桶，并统计每个桶的频数
        hist_pred = torch.histc(y_pred, bins=self.num_bins, min=min_val, max=max_val)
        hist_true = torch.histc(y_true, bins=self.num_bins, min=min_val, max=max_val)

        # 计算每个桶的累积分布函数（CDF）
        cdf_pred = torch.cumsum(hist_pred, dim=0) / bsz
        cdf_true = torch.cumsum(hist_true, dim=0) / bsz

        # 计算Wasserstein距离为两个CDF之间的L1距离的总和
        wasserstein_distance = torch.sum(torch.abs(cdf_pred - cdf_true))

        return wasserstein_distance


if __name__ == "__main__":
    from utils.config import rnn_arg
    import torch.optim as optim

    model = DagModel(rnn_arg)
    model.rand_sample()
    acti, archi = model.get_architecture()
    print(model.get_architecture())
    model.perturb(acti, archi)
    print(model.get_architecture())
    # input = torch.rand((64, 42, 12))
    # output, _ = model(input)
    # target = torch.rand((64, 1))
    # op = optim.Adam(model.parameters(), lr=1e-3)
    # c = nn.MSELoss()
    # loss = c(output, target)
    # print(loss.item())
    # loss.backward()
    # op.step()

    # print(gen_mask([35,100], 0.2))
