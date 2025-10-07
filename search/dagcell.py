import os
import random
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import RnnArgs


def get_func(idx):
    if idx == 0:
        return nn.Sigmoid()
    elif idx == 1:
        return nn.Tanh()
    elif idx == 2:
        return nn.ReLU()
    elif idx == 3:
        return nn.Hardswish()


def get_index(func):
    if isinstance(func, nn.Sigmoid):
        return 0
    elif isinstance(func, nn.Tanh):
        return 1
    elif isinstance(func, nn.ReLU):
        return 2
    elif isinstance(func, nn.Hardswish):
        return 3


class DagCell(nn.Module):
    def __init__(self, n_node, n_emb, n_hidden):
        super(DagCell, self).__init__()
        self.n_node = n_node
        self.n_hidden = n_hidden
        self.activation = []
        self.architecture = []
        self.actions = []
        self.output_node = []
        self.edges = {}
        self.n_train = {}
        for i in range(n_node):
            for j in range(i + 1, n_node):
                w = nn.Parameter(torch.Tensor(n_hidden, 2 * n_hidden))
                self.edges[f'Edge_{i}_{j}'] = w
                self.n_train[f'Edge_{i}_{j}'] = 0
        self.edges = nn.ParameterDict(self.edges)

        self.w0 = nn.Parameter(torch.Tensor(n_emb+n_hidden, 2 * n_hidden))

    def _compute_init_state(self, x, h_prev):
        xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self.w0), self.n_hidden, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def forward(self, x, h_prev):
        all_h = []
        h = self._compute_init_state(x, h_prev)
        all_h.append(h)
        output = torch.zeros_like(h_prev)
        for cur_index in range(self.n_node - 1):
            src_index, nonlinear, w = self.actions[cur_index]
            src_h = all_h[src_index]
            ch = torch.mm(src_h, w)
            c, h = torch.split(ch, self.n_hidden, dim=-1)
            c = F.sigmoid(c)
            h = c * nonlinear(h) + (1 - c) * src_h
            all_h.append(h)
        for node in self.output_node:
            output += all_h[node]
        output = output / len(self.output_node)
        return output, output

    def construct(self, activation, architecture):
        self.clear()
        self.activation = activation
        self.architecture = architecture
        output = set(self.architecture)
        for i in range(self.n_node):
            if i not in output:
                self.output_node.append(i)
        for i in range(self.n_node - 1):
            activation = get_func(self.activation[i])
            src_node = self.architecture[i]
            w = self.edges[f"Edge_{src_node}_{i + 1}"]
            self.actions.append((src_node, activation, w))
        # print(self.architecture, self.activation, self.output_node)

    def rand_sample(self):
        activation, architecture = [], []
        for i in range(self.n_node - 1):
            activation.append(random.randint(0, 3))
            architecture.append(random.randint(0, i))
        for index in range(len(architecture)):
            dst_node = index + 1
            src_node = architecture[index]
            self.n_train[f'Edge_{src_node}_{dst_node}'] += 1
        self.construct(activation, architecture)

    def clear(self):
        self.activation = []
        self.architecture = []
        self.output_node = []
        self.actions = []

    def perturb(self, activation, architecture, n_perturb):
        archi, acti = copy(architecture), copy(activation)
        low, high = max(0, n_perturb - len(archi) + 1), min(n_perturb, len(acti))
        perturb_acti = random.randint(low, high)
        perturb_archi = n_perturb - perturb_acti
        selected_archi = random.sample(range(1, len(archi)), perturb_archi)
        for index in selected_archi:
            cur, perv = archi[index], archi[index]
            while cur == perv:
                cur = random.randint(0, index)
            archi[index] = cur
        selected_acti = random.sample(range(len(acti)), perturb_acti)
        for index in selected_acti:
            cur, perv = acti[index], acti[index]
            while cur == perv:
                cur = random.randint(0, 3)
            acti[index] = cur
        self.construct(acti, archi)
