import os
import datetime
import argparse

from search.trainer import Trainer
from utils.config import encoder_arg, decoder_arg, train_arg, ctl_arg


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    layer_params = {name: p.numel() for name, p in model.named_parameters()}
    return total_params, layer_params

def main():
    parser = argparse.ArgumentParser(description="Run Neural Architecture Search.")

    # --- GPU and Logging --- 
    parser.add_argument('--visible_gpus', nargs='+', type=int, default=[1], help='Specify which GPU IDs to make visible.')
    parser.add_argument('--data_path', type=str, default=None, help='Path to log cell data. Defaults to a timestamped file in ./logs/')

    # --- Architecture Hyperparameters --- 
    parser.add_argument('--encoder_nodes', type=int, default=5, help='Number of nodes in the encoder DAG.')
    parser.add_argument('--encoder_hidden', type=int, default=128, help='Hidden size for the encoder RNN.')
    parser.add_argument('--decoder_nodes', type=int, default=3, help='Number of nodes in the decoder DAG.')
    parser.add_argument('--decoder_hidden', type=int, default=128, help='Hidden size for the decoder RNN.')

    # --- Training Hyperparameters --- 
    parser.add_argument('--controller_hid', type=int, default=200, help='Hidden size for the controller RNN.')
    parser.add_argument('--controller_lr', type=float, default=0.00035, help='Learning rate for the controller.')
    parser.add_argument('--train_lr', type=float, default=2e-3, help='Learning rate for the shared model.')
    parser.add_argument('--entropy_coeff', type=float, default=0.0001, help='Coefficient for the controller\'s entropy bonus.')
    parser.add_argument('--training_steps', type=int, default=20000, help='Number of training steps for the controller.')

    args = parser.parse_args()

    # --- Set up environment and configs --- 
    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_gpus))

    # Set up data path
    if args.data_path:
        train_arg.data_path = args.data_path
    else:
        start = datetime.datetime.now()
        stamp = str(int(start.timestamp()))
        train_arg.data_path = f'./logs/cell_data_{stamp}.json'

    # Override default configs with command-line arguments
    encoder_arg.n_node = args.encoder_nodes
    encoder_arg.n_hidden = args.encoder_hidden

    decoder_arg.n_node = args.decoder_nodes
    decoder_arg.n_input = encoder_arg.attention_output  # This is derived from a fixed value in RnnArgs
    decoder_arg.n_hidden = args.decoder_hidden

    ctl_arg.num_blocks = args.encoder_nodes + args.decoder_nodes
    ctl_arg.n_encoder = args.encoder_nodes
    ctl_arg.n_decoder = args.decoder_nodes
    ctl_arg.controller_hid = args.controller_hid
    ctl_arg.lr = args.controller_lr
    ctl_arg.entropy_coeff = args.entropy_coeff
    ctl_arg.training_step = args.training_steps

    train_arg.lr = args.train_lr

    # --- Start Training --- 
    print(f"Starting NAS with data log at: {train_arg.data_path}")
    trainer = Trainer(encoder_arg, decoder_arg, train_arg, ctl_arg)
    trainer.train()
    # print(count_parameters(trainer.model))

if __name__ == "__main__":
    main()
