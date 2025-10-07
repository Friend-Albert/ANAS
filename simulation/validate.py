import os
import sys
import warnings
import argparse

from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")
from scipy import stats
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')

import tqdm
import matplotlib.patches as mpatches


def showFig(dqn, nas, plot_dir):
    for col in ['delay', 'jitter']:
        fig, (ax) = plt.subplots(figsize=(8, 5))
        a_val = 1.
        colors = ['r', 'b', 'g', '#A020F0', '#8A2BE2', '#9933FA', '#8B4513', '#A0522D', '#CD853F']
        circ1 = mpatches.Patch(edgecolor=colors[0], alpha=a_val, linestyle='-', label='MAP - sim', fill=False)
        circ2 = mpatches.Patch(edgecolor=colors[1], alpha=a_val, linestyle='-', label='Poisson - sim', fill=False)
        circ3 = mpatches.Patch(edgecolor=colors[2], alpha=a_val, linestyle='-', label='Onoff - sim', fill=False)
        circ4 = mpatches.Patch(edgecolor=colors[3], alpha=a_val, linestyle='--', label='MAP - DQN', fill=False)
        circ5 = mpatches.Patch(edgecolor=colors[4], alpha=a_val, linestyle='--', label='Poisson - DQN', fill=False)
        circ6 = mpatches.Patch(edgecolor=colors[5], alpha=a_val, linestyle='--', label='Onoff - DQN', fill=False)
        circ7 = mpatches.Patch(edgecolor=colors[6], alpha=a_val, linestyle='--', label='MAP - ANAS', fill=False)
        circ8 = mpatches.Patch(edgecolor=colors[7], alpha=a_val, linestyle='--', label='Poisson - ANAS', fill=False)
        circ9 = mpatches.Patch(edgecolor=colors[8], alpha=a_val, linestyle='--', label='Onoff - ANAS', fill=False)

        for i in range(3):
            ti = ['MAP', 'Poisson', 'Onoff'][i]
            bins = np.histogram(np.hstack((nas[nas.tp == ti][col + '_sim'].values,
                                           nas[nas.tp == ti][col + '_pred'].values)), bins=100)[1]
            plt.hist(nas[nas.tp == ti][col + '_sim'].values, bins, density=True, color=colors[i], histtype='step',
                     linewidth=1.5)
            plt.hist(nas[nas.tp == ti][col + '_pred'].values, bins, density=True, color=colors[i+3], histtype='step',
                     linestyle='--', linewidth=0.75)
            plt.hist(dqn[dqn.tp == ti][col + '_pred'].values, bins, density=True, color=colors[i+6], histtype='step',
                     linestyle='--', linewidth=0.75)
        ax.legend(handles=[circ1, circ4, circ7, circ2, circ5, circ8, circ3, circ6, circ9], loc=1, fontsize=14)
        plt.xlabel(col.capitalize() + ' (sec)', fontsize=14)
        plt.ylabel('PDF', fontsize=14)
        plt.tick_params(labelsize=12)

        plt.savefig("{}/fattree_pdf_{}.png".format(plot_dir, col))

        fig, (ax) = plt.subplots(figsize=(8, 5))
        for i in range(3):
            ti = ['MAP', 'Poisson', 'Onoff'][i]
            res = stats.relfreq(dqn[dqn.tp == ti][col + '_sim'].values, numbins=100)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
            y = np.cumsum(res.frequency)
            plt.plot(x, y, color=colors[i], linewidth=1.5)
            res = stats.relfreq(dqn[dqn.tp == ti][col + '_pred'].values, numbins=100)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
            y = np.cumsum(res.frequency)
            plt.plot(x, y, color=colors[i+3], linestyle='--', linewidth=0.75)
            res = stats.relfreq(nas[nas.tp == ti][col + '_pred'].values, numbins=100)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
            y = np.cumsum(res.frequency)
            plt.plot(x, y, color=colors[i+6], linestyle='--', linewidth=0.75)
        plt.xlabel(col.capitalize() + ' (sec)', fontsize=14)
        plt.ylabel('CDF', fontsize=14)
        ax.legend(handles=[circ1, circ4, circ7, circ2, circ5, circ8, circ3, circ6, circ9], loc=4, fontsize=10)
        plt.tick_params(labelsize=12)

        plt.savefig("{}/fattree_cdf_{}.png".format(plot_dir, col))


def plot_regression(dqn_pred_org, nas_pred_org, gt_org, plot_dir, saved_figure_name):
    max_pos = max(max(max(dqn_pred_org), max(gt_org)), max(nas_pred_org))
    dqn_coef = round(np.corrcoef(gt_org, dqn_pred_org)[1, 0], 4)
    nas_coef = round(np.corrcoef(gt_org, nas_pred_org)[1, 0], 4)
    plt.scatter(gt_org, dqn_pred_org, s=[10 for _ in range(len(gt_org))], alpha=0.5, color='b',
                label="DQN \\rho = {}$".format(dqn_coef))
    plt.scatter(gt_org, nas_pred_org, s=[10 for _ in range(len(gt_org))], alpha=0.5, color='orange',
                label="ANAS \\rho = {}$".format(nas_coef))
    plt.plot([0, max_pos], [0, max_pos], linestyle="-.", color="red")
    plt.xlabel("GT")
    plt.ylabel("prediction result")
    plt.legend()

    plt.savefig("{}/{}".format(plot_dir, saved_figure_name))
    plt.clf()


def plot_ccdf(dqn_pred, nas_pred, gt, plot_dir, saved_figure_name):
    pr = np.cumsum(np.ones_like(dqn_pred))
    pr = pr / len(pr)
    ccdf = np.log(1 - pr + 1e-20)
    x_axis_max = max(max(max(dqn_pred), max(gt)), max(nas_pred))
    plt.plot(dqn_pred, ccdf, color="red", linestyle="-", label='DQN')
    plt.plot(nas_pred, ccdf, color="orange", linestyle="-", label='NAS')
    plt.plot(gt, ccdf, color="blue", linestyle="--", label='GT')
    plt.xlabel("Delay")
    plt.ylabel("log(1 - CDF)")
    plt.title("CCDF")
    plt.legend()
    plt.axis((0, x_axis_max, -10, 0))
    plt.savefig("{}/{}".format(plot_dir, saved_figure_name))

    plt.clf()


def plot_avgjitter_regression(dqn, nas, plot_dir):
    for col in ['delay', 'jitter']:
        for tp in ['MAP', 'Poisson', 'Onoff']:
            dqn_mean = dqn[dqn.tp == tp].groupby(["src_port", "path"])[[col + "_pred", col + "_sim"]].mean()
            nas_mean = nas[nas.tp == tp].groupby(["src_port", "path"])[[col + "_pred", col + "_sim"]].mean()
            gt = dqn_mean[col + "_sim"].values
            dqn_pred = dqn_mean[col + "_pred"].values
            nas_pred = nas_mean[col + "_pred"].values
            saved_name = "fattree_{}_{}_avgjitter.png".format(col, tp)
            plot_regression(dqn_pred, nas_pred, gt, plot_dir, saved_name)


def plot_delay_ccdf(dqn, nas, plot_dir):
    for tp in ['MAP', 'Poisson', 'Onoff']:
        dqn_mean = dqn[dqn.tp == tp].groupby(["src_port", "path"])[["delay_pred", "delay_sim"]].mean()
        nas_mean = nas[nas.tp == tp].groupby(["src_port", "path"])[["delay_pred", "delay_sim"]].mean()
        gt = dqn_mean["delay_sim"].values
        dqn_pred = dqn_mean["delay_pred"].values
        nas_pred = nas_mean["delay_pred"].values
        sorted_indices = np.argsort(gt)
        gt = gt[sorted_indices]
        saved_name = "fattree_delay_{}_.png".format(tp)
        plot_ccdf(dqn_pred, nas_pred, gt, plot_dir, saved_name)


def mergeTrace(identifier, tgs=['onoff', 'poisson', 'map']):
    result = pd.DataFrame()
    for traffic_pattern in tgs:
        for filename in ['rsim1']:
            # Corrected path to be relative to project root
            filepath = os.path.join('saved', identifier, 'fattree16', traffic_pattern, f'{filename}_pred.csv')
            try:
                t = pd.read_csv(filepath)
                t['delay_sim'] = t['dep_time'] - t['timestamp (sec)']
                t['delay_pred'] = t['etime'] - t['timestamp (sec)']
                t['fd'] = t['path'].apply(lambda x: len(x.split('-')))
                t['jitter_sim'] = t.groupby(['src_port', 'path'])['delay_sim'].diff().abs()
                t['jitter_pred'] = t.groupby(['src_port', 'path'])['delay_pred'].diff().abs()
                t['tp'] = traffic_pattern
                result = pd.concat([result, t], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: Trace file not found at {filepath}")
    return result


def cal_metrics(result, tgs=['map', 'poisson', 'onoff']):
    def percent99(x):
        return x.quantile(0.99)
    for tp in tgs:
        print(f'{tp:8s}', end=' ')
        for metric in ['delay', 'jitter']:
            nas_mean = result[result.tp == tp].groupby(["src_port", "path"])[[metric + "_pred", metric + "_sim"]].mean()
            gt = nas_mean[metric + "_sim"].values
            nas_pred = nas_mean[metric + "_pred"].values
            empty_array = np.zeros((len(gt),))
            nas_w1_dist = round(wasserstein_distance(gt, nas_pred) / wasserstein_distance(gt, empty_array), 3)
            nas_filtered = result[result.tp == tp].groupby(["src_port", "path"])[[metric + "_pred", metric + "_sim"]].agg(
                percent99)
            gt_p99 = nas_filtered[metric + "_sim"].values
            nas_p99 = nas_filtered[metric + "_pred"].values
            empty_array = np.zeros((len(gt_p99),))
            nas_w1_p99 = round(wasserstein_distance(gt_p99, nas_p99) / wasserstein_distance(gt_p99, empty_array), 3)
            print(f"{nas_w1_dist} {nas_w1_p99}", end=' ')
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate simulation results and generate plots.")
    parser.add_argument('--identifier', type=str, required=True, help='Identifier for the model/simulation run.')
    parser.add_argument('--traffic_patterns', nargs='+', default=['tmp'], help='List of traffic patterns to analyze.')
    args = parser.parse_args()

    plot_dir = f"saved/{args.identifier}/fattree_plot"

    if not os.path.exists(f"saved/{args.identifier}"):
        print("Invalid identifier! Base directory not found.")
        sys.exit(0)

    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Processing identifier: {args.identifier}")
    result = mergeTrace(args.identifier, args.traffic_patterns)
    
    if result.empty:
        print("No data found to validate. Exiting.")
        sys.exit(0)

    result = result.dropna()
    cal_metrics(result, args.traffic_patterns)
    print(f"Plots will be saved in: {plot_dir}")

if __name__ == "__main__":
    main()
