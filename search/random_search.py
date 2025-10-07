import json
import os
import time
import datetime
import argparse

import torch
from numpy import argmin
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import RnnArgs, TrainArgs, encoder_arg, decoder_arg, train_arg
from search.dagmodel import DagModel
from utils.Dataset import H5Dataset


class Trainer:
    def __init__(self, encoder_args: RnnArgs, decoder_args: RnnArgs, train_args: TrainArgs):
        torch.manual_seed(train_args.seed)

        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.train_args = train_args
        self.n_sample = 400
        self.idx_sample = 0
        self.best_loss = 100.
        self.best_acti, self.best_archi = [], []

        torch.manual_seed(train_args.seed)
        self.device = torch.device("cuda") if train_args.cuda else torch.device("cpu")

        train_dataset = H5Dataset(train_args.base_dir, mode="train")
        valid_dataset = H5Dataset(train_args.base_dir, mode="sampled_valid")
        self.train_data = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)
        self.val_data = DataLoader(valid_dataset, batch_size=train_args.test_bsz, shuffle=False)
        self.model = DagModel(encoder_args, decoder_args).to(self.device)
        self.model.resample()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_args.lr,
        )

        print("Train data len:", len(train_dataset))
        print("Valid data len:", len(valid_dataset))

    def train(self):
        start = datetime.datetime.now()
        print(f"Random Search Start at {start}")
        self._train_shared(self.train_args.collect_interval)
        print(f"Final Result: loss={self.best_loss}, activation={self.best_acti}, architecture{self.best_archi}")
        self.model.construct(self.best_acti, self.best_archi)
        self.model.save("./logs/random_search.json")
        end = datetime.datetime.now()
        print(f'Finished in {end}, used {end - start}', "=")


    def _train_shared(self, interval):
        self.model.train()
        total_loss = 0.
        is_end, start_time = False, time.time()
        while True:
            for batch, (data, targets) in enumerate(self.train_data, start=1):
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if batch % interval == 0:
                    cur_loss = total_loss / interval
                    elapsed = time.time() - start_time
                    val_loss = self._evaluate_all(self.val_data)
                    print('| n_sample {:3d} | ms/batch {:5.2f} | loss {:7.6f} | val_loss {:7.6f}'.format(
                        self.idx_sample, elapsed * 1000 / interval, cur_loss, val_loss))
                    self.idx_sample += 1
                    start_time = time.time()
                    total_loss = 0.
                    self.model.resample()
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_acti, self.best_archi = self.model.get_architecture()
                        print(f"New Best: loss:{val_loss:7.6f}, activation={self.best_acti}, architecture={self.best_archi}")
                    if self.idx_sample > self.n_sample:
                        return

    def _evaluate_all(self, data_source):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for batch, (data, targets) in enumerate(data_source):
                output, _ = self.model(data)
                total_loss += self.criterion(output, targets).item()
        self.model.train()
        return total_loss / len(data_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Search for Neural Architecture Search.")

    # --- GPU and Logging ---
    parser.add_argument('--visible_gpus', nargs='+', type=int, default=[0], help='Specify which GPU IDs to make visible.')

    # --- Architecture Hyperparameters ---
    parser.add_argument('--encoder_nodes', type=int, default=6, help='Number of nodes in the encoder DAG.')
    parser.add_argument('--encoder_hidden', type=int, default=200, help='Hidden size for the encoder RNN.')
    parser.add_argument('--decoder_nodes', type=int, default=6, help='Number of nodes in the decoder DAG.')
    parser.add_argument('--decoder_hidden', type=int, default=100, help='Hidden size for the decoder RNN.')

    args = parser.parse_args()

    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_gpus))

    # Override default configs with command-line arguments
    encoder_arg.n_node = args.encoder_nodes
    encoder_arg.n_hidden = args.encoder_hidden
    decoder_arg.n_node = args.decoder_nodes
    decoder_arg.n_input = encoder_arg.attention_output # This is derived from a fixed value in RnnArgs
    decoder_arg.n_hidden = args.decoder_hidden

    trainer = Trainer(encoder_arg, decoder_arg, train_arg)
    trainer.train()
