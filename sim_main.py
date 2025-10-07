import os
import sys
import multiprocessing as mp
import pandas as pd
import torch
import argparse

from simulation.fattree import fattree
from simulation.validate import cal_metrics


def run_simulation(args):
    """Runs the network simulation based on the provided configuration."""
    print("--- Starting Network Simulation ---")
    
    gpu_number = args.gpu_number
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running on CPU.", file=sys.stderr)
        gpu_number = 1
    elif torch.cuda.device_count() < gpu_number:
        print(f"Warning: Requested {gpu_number} GPUs, but only {torch.cuda.device_count()} are available.", file=sys.stderr)
        gpu_number = torch.cuda.device_count()

    for traffic in args.traffic_patterns:
        for index in args.file_indices:
            print(f"Running Simulation: Topology={args.topology}, Traffic={traffic}, Index={index}, Model={args.model_identifier}")
            
            if args.topology == "fattree":
                sim_instance = fattree(
                    file_idx=index,
                    k=args.k_val,
                    traffic_pattern=traffic,
                    base_dir=args.base_dir,
                    model_identifier=args.model_identifier
                )
            else:
                print(f"Error: Unknown topology type '{args.topology}'. Only 'fattree' is supported.", file=sys.stderr)
                sys.exit(1)

            sim_instance.run_parallel(gpu_number=gpu_number)
            print(f"Finished simulation for {traffic} rsim{index+1}.")
            print("-" * 20)
    print("--- Network Simulation Phase Complete ---")

def merge_trace_dynamic(identifier, tgs, topology_type, topology_size_params, file_indices):
    """Merges trace files from simulation runs into a single DataFrame."""
    result = pd.DataFrame()
    for traffic_pattern in tgs:
        for i in file_indices:
            filename = f'rsim{i+1}'
            
            if topology_type == "fattree":
                size_str = f'fattree{int(topology_size_params["K"]**3/4)}'
            else:
                print(f"Error: Unknown topology type '{topology_type}' for merging traces.", file=sys.stderr)
                sys.exit(1)

            # Use os.path.join for robust path construction
            filepath = os.path.join('saved', identifier, size_str, traffic_pattern, f'{filename}_pred.csv')
            if not os.path.exists(filepath):
                print(f"Warning: File not found, skipping: {filepath}", file=sys.stderr)
                continue
            t = pd.read_csv(filepath)
            print(filepath)
            t['delay_sim'] = t['dep_time'] - t['timestamp (sec)']
            t['delay_pred'] = t['etime'] - t['timestamp (sec)']
            t['jitter_sim'] = t.groupby(['src_port', 'path'])['delay_sim'].diff().abs()
            t['jitter_pred'] = t.groupby(['src_port', 'path'])['delay_pred'].diff().abs()
            t['tp'] = traffic_pattern
            result = pd.concat([result, t], ignore_index=True)
    return result

def run_validation(args):
    """Runs the validation phase based on the provided configuration."""
    print(f"\n--- Starting Validation for Model: {args.model_identifier} ---")
    
    topology_size_params = {}
    if args.topology == "fattree":
        topology_size_params["K"] = args.k_val
        size_str = f'fattree{int(args.k_val**3/4)}'
    else:
        print(f"Error: Unknown topology type '{args.topology}' for validation.", file=sys.stderr)
        sys.exit(1)

    # Use os.path.join for robust path construction
    output_dir = os.path.join('saved', args.model_identifier, size_str)
    if not os.path.exists(output_dir):
        print(f"Validation Error: Output directory not found at {output_dir}", file=sys.stderr)
        print("Please run the simulation first or ensure the model_identifier is correct.", file=sys.stderr)
        sys.exit(1)
        
    try:
        result_df = merge_trace_dynamic(args.model_identifier, args.traffic_patterns, 
                                        args.topology, topology_size_params, args.file_indices)
        
        if result_df.empty:
             print("Validation Error: No data found to validate. Exiting.", file=sys.stderr)
             sys.exit(1)

        result_df.dropna(inplace=True)
        
        print("\nValidation Metrics (Wasserstein Distance):")
        print(f"{'Traffic':<15} {'Delay (Mean)':<15} {'Jitter (Mean)':<15}")
        print("-" * 50)
        cal_metrics(result_df, tgs=args.traffic_patterns)
    except Exception as e:
        print(f"An error occurred during validation: {e}", file=sys.stderr)
    
    print("--- Validation Phase Complete ---")

def main():
    """
    Main function to run network simulation and validation based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run network simulation and validation.")
    
    # --- Simulation Parameters ---
    parser.add_argument('--topology', type=str, default='fattree', help='Network topology to simulate. Currently only "fattree" is supported.')
    parser.add_argument('--k_val', type=int, default=4, help='K-value for the Fat-Tree topology.')
    parser.add_argument('--traffic_patterns', nargs='+', default=['tmp'], help='List of traffic patterns to simulate.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Identifier for the model to be evaluated.')
    parser.add_argument('--gpu_number', type=int, default=4, help='Number of GPUs to use for simulation.')
    parser.add_argument('--file_indices', nargs='+', type=int, default=[0], help='List of file indices for simulation runs.')
    parser.add_argument('--base_dir', type=str, default='./data', help='Base directory for traffic data.')
    parser.add_argument('--visible_gpus', nargs='+', type=int, default=None, help='Specify which GPU IDs to make visible to the script (e.g., 0 1 2 3).')

    # --- Execution Control ---
    parser.add_argument('--skip_simulation', action='store_true', help='Set to skip the simulation phase.')
    parser.add_argument('--skip_validation', action='store_true', help='Set to skip the validation phase.')

    args = parser.parse_args()

    # Set visible GPUs if specified
    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_gpus))

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Simulation Phase ---
    if not args.skip_simulation:
        run_simulation(args)

    # --- Validation Phase ---
    if not args.skip_validation:
        run_validation(args)

if __name__ == "__main__":
    main()

