import json
import time
import datetime

import torch
from numpy import argmin
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import RnnArgs, TrainArgs, ControllerArgs
from search.dagmodel import DagModel
from search.controller import Controller
from search.predictor import Predictor
from utils.Dataset import H5Dataset


def print_separate(str, separator="-"):
    print(separator * 89)
    print(str)
    print(separator * 89)


def count_parameters(model):
    total_params = sum(p.numel() for k, p in model.named_parameters() if "Edge_0" in k or 'coder.edges' not in k)
    layer_params = {name: p.numel() for name, p in model.named_parameters() if
                    "Edge_0" in name or 'coder.edges' not in name}
    return total_params, layer_params


class Trainer:
    def __init__(self, encoder_args: RnnArgs, decoder_args: RnnArgs, train_args: TrainArgs, ctl_args: ControllerArgs):
        torch.manual_seed(train_args.seed)

        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.train_args = train_args
        self.ctl_args = ctl_args

        torch.manual_seed(train_args.seed)
        self.device = torch.device("cuda") if train_args.cuda else torch.device("cpu")

        self.model = DagModel(encoder_args, decoder_args).to(self.device)
        train_dataset = H5Dataset(train_args.base_dir, mode="train")
        valid_dataset = H5Dataset(train_args.base_dir, mode="sampled_valid")
        self.train_data = DataLoader(train_dataset, batch_size=512, shuffle=True)
        self.val_data = DataLoader(valid_dataset, batch_size=512, shuffle=False)
        self.predictor = Predictor(train_args.data_path, encoder_args.n_node, decoder_args.n_node)
        self.agent = Controller(ctl_args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_args.lr,
        )

        print("Device:", next(self.model.parameters()).device)
        print("Train data len:", len(train_dataset))
        print("Valid data len:", len(valid_dataset))
        print(f"Activate Params: {count_parameters(self.model)[0]}")

    def train(self):
        start = datetime.datetime.now()

        print(f"ANAS Start at {start}")
        if not self.train_args.isENAS:
            print_separate('Training Shared Model', "=")
            if self.train_args.enable_warmup:
                print_separate("Warm Up Phase")
                is_end, n_sample = False, 0
                while not is_end:
                    n_sample, is_end = \
                        self._train_shared(self.train_args.warmup_interval, self.train_args.warmup_sample, n_sample, True)
            print_separate("Collect data Phase")
            is_end, n_sample = False, 0
            while not is_end:
                n_sample, is_end = \
                    self._train_shared(self.train_args.collect_interval, self.train_args.collect_sample, n_sample, False)
            print_separate("Training Predictor Phase")
            self.predictor.train()
            print_separate("Training Agent Phase")
            self._train_controller()
            acti, archi = self._derive()
            print(f"Best Architecture is activation={acti}, architecture={archi}")
            self.model.construct(acti, archi)
            stamp = str(int(start.timestamp()))
            self.model.save(f"./logs/best_{stamp}.json")
        else:
            for i in range(self.train_args.collect_sample):
                acti, archi = self.agent.sample(detail=False)
                self.model.construct(acti, archi)
                self.enas_train(i, self.train_args.collect_interval)
                self.enas_controller(2000)
            self.enas_derive(100)
        end = datetime.datetime.now()
        print_separate(f'Finished in {end}, used {end - start}', "=")

    def _train_shared(self, interval, max_sample, n_sample, is_warmup):
        self.model.train()
        total_loss = 0.
        is_end, start_time = False, time.time()
        for batch, (data, targets) in enumerate(self.train_data, start=1):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch % interval == 0:
                cur_loss = total_loss / interval
                elapsed = time.time() - start_time
                val_loss = self._evaluate_all(self.val_data)
                print('| n_sample {:3d} | ms/batch {:5.2f} | loss {:7.6f} | val_loss {:7.6f}'.format(
                    n_sample, elapsed * 1000 / interval, cur_loss, val_loss))
                self.model.n_norm = 0
                if not is_warmup:
                    self._collect(800)
                self.model.resample()
                n_sample += 1
                start_time = time.time()
                total_loss = 0.
                if n_sample > max_sample:
                    is_end = True
                    break
        return n_sample, is_end

    def _collect(self, n):
        all_data = []
        std_acti, std_archi = self.model.get_architecture()
        for i in range(n):
            loss = self._evaluate(self.val_data)
            rectify_loss = loss * self.model.rectify_factor
            activation, architecture = self.model.get_architecture()
            data = {'cell': activation + architecture,
                    'loss': round(loss, 7),
                    'ret': round(rectify_loss, 7)}
            all_data.append(data)
            if self.train_args.use_sim:
                self.model.perturb(std_acti, std_archi)
            else:
                self.model.resample()
        if self.train_args.use_sim:
            all_data = sorted(all_data, key=lambda x: x.get('loss', float('inf')))[:int(0.9 * n)]
        with open(self.train_args.data_path, 'a') as f:
            for data in all_data:
                json.dump(data, f)
                f.write('\n')

    def _evaluate(self, data_source):
        self.model.eval()
        with torch.no_grad():
            for batch, (data, targets) in enumerate(data_source, start=1):
                output = self.model(data)
                loss = self.criterion(output, targets).item()
                break
        return loss

    def _train_controller(self):
        total_r, best_r = 0., 0.
        start = time.time()
        for i in range(1, self.ctl_args.training_step + 1):
            acti, archi, log_probs, entropies = self.agent.sample(detail=True)
            cell = acti + archi
            loss = self.predictor.predict(cell)
            if loss < 5e-4:
                self.model.construct(acti, archi)
                loss = self._evaluate(self.val_data)
            raw_r, r = self.get_reward(loss, entropies)
            total_r += raw_r
            self.agent.step(r, log_probs)
            if i % 500 == 0:
                t = time.time() - start
                avg_r = total_r / 500
                print(f"| step {i} | avg reward {avg_r:5.4f} | time {t:4.2f}")
                self.agent.reset_history()
                total_r = 0.
                start = time.time()
                if best_r < avg_r:
                    best_r = avg_r
                    with open(self.train_args.model_path + 'controller.pt', 'wb') as f:
                        torch.save(self.agent, f)

    def get_reward(self, loss, entropies):
        raw_reward = self.ctl_args.reward_c / loss
        rewards = raw_reward + self.ctl_args.entropy_coeff * entropies
        return raw_reward, rewards

    def _derive(self):
        derive_acti, derive_archi, derive_loss = [], [], []
        with open(self.train_args.model_path + 'controller.pt', 'rb') as f:
            self.agent = torch.load(f)
        for i in range(self.ctl_args.derive_step):
            with torch.no_grad():
                acti, archi = self.agent.sample(detail=False)
                print(acti, archi)
                derive_acti.append(acti), derive_archi.append(archi)
        for i in range(self.ctl_args.derive_step):
            self.model.construct(derive_acti[i], derive_archi[i])
            self.model.train()
            for batch, (data, targets) in enumerate(self.train_data, start=1):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                if batch == self.train_args.collect_interval:
                    break

            loss = self._evaluate_all(self.val_data)
            derive_loss.append(loss)
            print('| index {:3d} | loss {:7.6f}'.format(i, loss))
        best_idx = argmin(derive_loss)
        print(f'The optimal architecture proposed by the controller is: '
              f'activation={derive_acti[best_idx]} architecture={derive_archi[best_idx]}')
        return derive_acti[best_idx], derive_archi[best_idx]

    def _evaluate_all(self, data_source):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for batch, (data, targets) in enumerate(data_source):
                output = self.model(data)
                total_loss += self.criterion(output, targets).item()
        return total_loss / len(data_source)

    def enas_train(self, n_sample, interval):
        self.model.train()
        total_loss = 0.
        is_end, start_time = False, time.time()
        for batch, (data, targets) in enumerate(self.train_data, start=1):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch % interval == 0:
                cur_loss = total_loss / interval
                elapsed = time.time() - start_time
                val_loss = self._evaluate_all(self.val_data)
                print('| n_sample {:3d} | ms/batch {:5.2f} | loss {:7.6f} | val_loss {:7.6f} |'.format(
                    n_sample, elapsed * 1000 / interval, cur_loss, val_loss), end=' ')
                return

    def enas_controller(self, step):
        total_r, best_r = 0., 0.
        start = time.time()
        self.agent.train()
        for i in range(1, step + 1):
            acti, archi, log_probs, entropies = self.agent.sample(detail=True)
            self.model.construct(acti, archi)
            loss = self._evaluate(self.val_data)
            raw_r, r = self.get_reward(loss, entropies)
            total_r += raw_r
            self.agent.step(r, log_probs)
        t = time.time() - start
        avg_r = total_r / step
        print(f"avg reward {avg_r:5.4f} | time {t:4.2f} |")
        self.agent.reset_history()

    def enas_derive(self, n):
        derive_acti, derive_archi, derive_loss = [], [], []
        self.agent.eval()
        for i in range(n):
            with torch.no_grad():
                acti, archi = self.agent.sample(detail=False)
                print(acti, archi)
                derive_acti.append(acti), derive_archi.append(archi)
        for i in range(n):
            self.model.construct(derive_acti[i], derive_archi[i])
            self.model.train()
            loss = self._evaluate_all(self.val_data)
            derive_loss.append(loss)
            print('| index {:2d} | loss {:7.6f}'.format(i, loss))
        best_idx = argmin(derive_loss)
        print(f'The optimal architecture proposed by the controller is: '
              f'activation={derive_acti[best_idx]} architecture={derive_archi[best_idx]}')
        self.model.construct(derive_acti[best_idx], derive_archi[best_idx])
        self.model.save(f"./logs/best.json")
        return
