import datetime
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance

from utils.Dataset import H5Dataset

class WassersteinLoss(nn.Module):
    def __init__(self, num_bins):
        super(WassersteinLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        bsz = y_pred.shape[0]
        min_val = min(torch.min(y_pred).item(), torch.min(y_true).item())
        max_val = max(torch.max(y_pred).item(), torch.max(y_true).item())

        hist_pred = torch.histc(y_pred, bins=self.num_bins, min=min_val, max=max_val) / bsz
        hist_true = torch.histc(y_true, bins=self.num_bins, min=min_val, max=max_val) / bsz

        cdf_pred = torch.cumsum(hist_pred, dim=0)
        cdf_true = torch.cumsum(hist_true, dim=0)

        wasserstein_distance = torch.sum(torch.abs(cdf_pred - cdf_true))

        return wasserstein_distance

def main():
    parser = argparse.ArgumentParser(description="Validate a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pt).')
    parser.add_argument('--base_dir', type=str, default='./data', help='Base directory for the dataset.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for the validation loader.')
    parser.add_argument('--visible_gpus', nargs='+', type=int, default=[0], help='Specify which GPU IDs to make visible.')

    args = parser.parse_args()

    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_gpus))

    print(f"Loading model from: {args.model_path}")
    try:
        with open(args.model_path, 'rb') as f:
            model = torch.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading dataset from: {args.base_dir}")
    valid_dataset = H5Dataset(args.base_dir, mode="sampled_valid")
    validation_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f'Validation batches: {len(validation_loader)}')

    model.eval()
    mse_loss = 0.0
    loss_func1 = torch.nn.MSELoss()
    eval_epoch_batch_num = len(validation_loader.dataset)
    gt = []
    output = []
    with torch.no_grad():
        for idx, (batch_x, batch_y) in enumerate(validation_loader):
            out = model(batch_x)
            loss1 = loss_func1(out, batch_y)
            mse_loss += loss1.item() * batch_y.shape[0]
            gt += batch_y.reshape(-1).tolist()
            output += out.reshape(-1).tolist()

    if eval_epoch_batch_num > 0:
        mse_avg_loss = mse_loss / eval_epoch_batch_num
        print("Eval avg mse loss: {} | Eval avg W loss: {}".format(mse_avg_loss, wasserstein_distance(gt, output) / wasserstein_distance(gt, [0]*len(gt))))

        # p99 calculation
        if len(gt) > 100:
            gt_p99 = gt[-int(0.01*len(gt))]
            output_p99 = output[-int(0.01*len(output))]
            p99_wasserstein = wasserstein_distance([gt_p99], [output_p99])
            print(f"p99 Eval wasserstein distance (scipy): {p99_wasserstein}")
    else:
        print("Validation dataset is empty. Cannot compute metrics.")

if __name__ == "__main__":
    main()
