import os, sys
import pandas as pd
import numpy as np
import time
import copy
import math

import multiprocessing as mp
from multiprocessing import Process, Manager, connection

import torch

from tf_impl.minmaxscaler import load_scaler
from tf_impl.trafficindicator import feaTure
from tf_impl.error_corr import get_error_dis, error_correction

from utils.config import BaseConfig, ModelConfig

import warnings

warnings.filterwarnings("ignore")


class fattree:

    def __init__(self,
                 file_idx,
                 k=4,
                 traffic_pattern="poisson",
                 base_dir="./data",
                 model_identifier="it_10",
                 ):

        self.config = BaseConfig()
        self.model_config = ModelConfig()
        self.model_k = k

        self.base_dir = base_dir
        self.model_identifier = model_identifier

        self._generate_fattree_topology(k)

        self.input_file = '{}/fattree{}/{}/rsim{}'.format(self.base_dir, int(self.k**3/4), traffic_pattern, file_idx + 1)
        self.output_file = 'saved/{}/fattree{}/{}/rsim{}_pred.csv'.format(model_identifier, int(self.k**3/4), traffic_pattern, file_idx + 1)
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        flow = pd.read_csv('{}.csv'.format(self.input_file)).reset_index(drop=True)
        flow['src_hub'] = flow['src_pc'] // (self.k // 2) + self.edge_start_id
        flow['src_port'] = flow['src_pc'] % (self.k // 2)

        self.flow = self.init_flow(flow)
        print("flow loading and initialization done ...")
        print("flow data shape:", self.flow.shape)

        if self.model_config.error_correction:
            self.ERR, self.BINS = get_error_dis(self.config, self.model_config)

    def _generate_fattree_topology(self, k):
        if k % 2 != 0:
            raise ValueError("k must be an even number for Fat-Tree topology.")

        self.k = k
        self.ports_per_switch = k
        self.num_pods = k
        
        self.num_core = (k // 2) ** 2
        self.num_agg = k * (k // 2)
        self.num_edge = k * (k // 2)
        self.total_switches = self.num_core + self.num_agg + self.num_edge
        self.num_hosts = k * (k // 2) * (k // 2)

        self.core_start_id = 0
        self.core_end_id = self.num_core - 1
        self.agg_start_id = self.num_core
        self.agg_end_id = self.num_core + self.num_agg - 1
        self.edge_start_id = self.num_core + self.num_agg
        self.edge_end_id = self.num_core + self.num_agg + self.num_edge - 1

        self.port_connection_matrix = [[-1] * self.ports_per_switch for _ in range(self.total_switches)]
        self.port_connection = dict()

        for pod in range(k):
            for i in range(k // 2):
                edge_switch_id = self.edge_start_id + pod * (k // 2) + i
                for port in range(k // 2, k):
                    agg_switch_id = self.agg_start_id + pod * (k // 2) + (port - k // 2)
                    self.port_connection_matrix[edge_switch_id][port] = agg_switch_id
                    self.port_connection_matrix[agg_switch_id][i] = edge_switch_id

        for pod in range(k):
            for i in range(k // 2):
                agg_switch_id = self.agg_start_id + pod * (k // 2) + i
                for port in range(k // 2, k):
                    core_switch_id = self.core_start_id + i * (k // 2) + (port - k // 2)
                    self.port_connection_matrix[agg_switch_id][port] = core_switch_id
                    self.port_connection_matrix[core_switch_id][pod] = agg_switch_id
        
        self.config.no_of_port = k
        self.saved_model = "saved/{}/saved_model/best_model.pt".format(self.model_identifier)

        self.model = None
        self.model_path = self.saved_model

    def init_flow(self, flow):
        flow['dep_time'] = flow['etime']
        flow['etime'] = flow['timestamp (sec)']
        flow['cur_hub'] = flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[0]))
        flow['cur_port'] = flow['path'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
        flow['status'] = 0
        return flow

    def parse_topo(self) -> None:
        for from_node, to_list in enumerate(self.port_connection_matrix):
            for to_node, to_port in enumerate(to_list):
                if from_node not in self.port_connection:
                    self.port_connection[from_node] = dict()
                if to_port >= 0:
                    self.port_connection[from_node][to_port] = to_node

    def model_input(self, df, layer):
        if df.empty:
            return pd.DataFrame(columns=self.fet_cols)
        dt = df.copy()
        dt['timestamp (sec)'] = dt['etime']
        dt['src'] = dt[['src_port', 'status', 'path']].apply(self.get_src, axis=1).values
        dt['dst'] = dt['cur_port']

        self.add_egress_feature(dt)
        self.add_ingress_fetature(dt)
        dt = dt.fillna(method='ffill').fillna(method='bfill').fillna(0.)
        return dt[self.fet_cols].copy()

    def get_src(self, x):
        src_port, status, path = x['src_port'], x['status'], x['path']
        if status == 0:
            return src_port
        else:
            last_hub = int(path.split('-')[status - 1].split('_')[0])
            cur_hub = int(path.split('-')[status].split('_')[0])
            if self.core_start_id <= last_hub <= self.core_end_id:
                last_layer = 'core'
            elif self.agg_start_id <= last_hub <= self.agg_end_id:
                last_layer = 'agg'
            else:
                last_layer = 'edge'
            if self.core_start_id <= cur_hub <= self.core_end_id:
                cur_layer = 'core'
            elif self.agg_start_id <= cur_hub <= self.agg_end_id:
                cur_layer = 'agg'
            else:
                cur_layer = 'edge'
            if last_layer == 'core':
                pod = self.port_connection_matrix[last_hub].index(cur_hub)
                return pod
            elif last_layer == 'agg':
                return self.port_connection_matrix[last_hub].index(cur_hub)
            else:
                return self.port_connection_matrix[last_hub].index(cur_hub)

    def load_scaler(self):
        self.x_MIN, self.x_MAX, self.y_MIN, self.y_MAX, self.fet_cols, _ = load_scaler(
            "{}/_scaler".format(self.base_dir))

    def load_model(self, return_model=False):
        import sys
        if 'utils.config' in sys.modules:
            sys.modules['config'] = sys.modules['utils.config']
        underlying_model = torch.load(self.model_path)
        underlying_model.eval()
        self.model = underlying_model
        if return_model:
            return copy.deepcopy(self.model)

    def add_ingress_fetature(self, df):
        df['inter_arr_sys'] = df['timestamp (sec)'].diff()

    def add_egress_feature(self, df):
        ins = feaTure(df,
                      self.config.no_of_port,
                      self.config.no_of_buffer,
                      self.config.window,
                      self.config.ser_rate)
        Count_dst_SET, LOAD = ins.getCount()
        for i in range(self.config.no_of_port):
            df['port_load%i' % i] = LOAD[i]
        for i in range(self.config.no_of_port):
            for j in range(self.config.no_of_buffer):
                df['load_dst%i_%i' % (i, j)] = np.array(Count_dst_SET[(i, j)])

    def timeseries(self, sample):
        dim0 = sample.shape[0] - self.config.TIME_STEPS + 1
        if dim0 <= 0:
            return np.empty((0, self.config.TIME_STEPS, sample.shape[1]))
        dim1 = sample.shape[1]
        x = np.zeros((dim0, self.config.TIME_STEPS, dim1))
        for i in range(dim0):
            x[i] = sample[i:self.config.TIME_STEPS + i]
        return x

    def flows_to_port(self):
        self.IN = dict()
        for i in range(self.total_switches):
            self.IN['DataFrame{}'.format(i)] = self.flow[self.flow.src_hub == i].copy().sort_values(
                'etime').reset_index(drop=True)

    def predict(self, data, model, device, batch_size=2048):
        if data.shape[0] == 0:
            return np.array([])
        out_list = []
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            data_len = len(data)
            for start_idx in range(0, data_len, batch_size):
                end_idx = start_idx + batch_size if start_idx + batch_size <= data_len else data_len
                batch = data[start_idx:end_idx].to(device)
                out = model(batch)
                out_list.append(out.cpu().flatten())
        return_val = torch.cat(out_list)
        assert len(return_val) == len(data)
        return return_val.numpy()

    def infer(self, flow, layer, model, device):
        input_dt = self.model_input(flow, layer)
        input_dt = (input_dt - self.x_MIN) / (self.x_MAX - self.x_MIN + 1e-9)
        x_input = self.timeseries(input_dt.values)
        y_pred = self.predict(x_input, model, device)
        y_pred = self.y_MIN[-1] + y_pred * (self.y_MAX[-1] - self.y_MIN[-1])
        flow = flow.iloc[self.config.TIME_STEPS - 1:]
        if self.model_config.error_correction:
            flow['delay'] = y_pred
            flow = error_correction(flow, self.config, self.ERR, self.BINS)
            y_pred = [max(x, float(y) / self.config.ser_rate) for x, y in
                      zip(flow.delay.values, flow['pkt len (byte)'].values)]
            flow['etime'] = flow['etime'] + y_pred
            flow.drop(['delay', 'bins'], axis=1, inplace=True)
        else:
            y_pred = [max(x, float(y) / self.config.ser_rate) for x, y in zip(y_pred, flow['pkt len (byte)'].values)]
            flow['etime'] = flow['etime'] + y_pred
        return flow

    def infer_multi_nodes(self, flow_dict, layer, model, device):
        flow_len_dict = dict()
        all_input_list = []
        node_order = []

        for node_id, flow in flow_dict.items():
            if not flow.empty:
                input_dt = self.model_input(flow, layer)
                normalized_dt = (input_dt - self.x_MIN) / (self.x_MAX - self.x_MIN + 1e-9)
                x_input = self.timeseries(normalized_dt.values)
                flow_len_dict[node_id] = len(x_input)
                all_input_list.append(x_input)
                node_order.append(node_id)

        if not all_input_list:
            return dict()

        x_input_all = np.concatenate(all_input_list)
        y_pred_all = self.predict(x_input_all, model, device)
        y_pred_all = self.y_MIN[-1] + y_pred_all * (self.y_MAX[-1] - self.y_MIN[-1])

        return_flow_dict = dict()
        start_pos = 0
        for i, node_id in enumerate(node_order):
            flow = flow_dict[node_id]
            flow_len = flow_len_dict[node_id]
            flow = flow.iloc[self.config.TIME_STEPS - 1:]
            end_pos = start_pos + flow_len
            y_pred = y_pred_all[start_pos:end_pos]
            start_pos = end_pos

            if self.model_config.error_correction:
                flow['delay'] = y_pred
                flow = error_correction(flow, self.config, self.ERR, self.BINS)
                y_pred = [max(x, float(y) / self.config.ser_rate) for x, y in
                          zip(flow.delay.values, flow['pkt len (byte)'].values)]
                flow['etime'] = flow['etime'] + y_pred
                flow.drop(['delay', 'bins'], axis=1, inplace=True)
            else:
                y_pred = [max(x, float(y) / self.config.ser_rate) for x, y in
                          zip(y_pred, flow['pkt len (byte)'].values)]
                flow['etime'] = flow['etime'] + y_pred
            return_flow_dict[node_id] = flow

        return return_flow_dict

    def link_info_upd(self, link):
        link['status'] += 1
        link['cur_hub'] = link[['status', 'path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[0]), axis=1)
        link['cur_port'] = link[['status', 'path']].apply(lambda x: int(x[1].split('-')[x[0]].split('_')[1]), axis=1)
        return link

    def trace_upd(self, layer, i, my_result, my_link, model, device):
        trace_in = self.IN['DataFrame{}'.format(i)].copy()
        for c in range(self.total_switches):
            link_data = my_link[(c, i)]
            if link_data.shape[0] > 0:
                trace_in = trace_in.append(link_data, ignore_index=True)
        trace_in = trace_in.sort_values('etime').reset_index(drop=True)
        trace_out = self.infer(trace_in, layer, model, device)
        for connected_port, connected_node in enumerate(self.port_connection_matrix[i]):
            if connected_node > -1:
                eport_flow = trace_out[trace_out.cur_port == connected_port]
                if eport_flow.shape[0] > 0:
                    my_link[(i, connected_node)] = self.link_info_upd(eport_flow)
        if self.edge_start_id <= i <= self.edge_end_id:
            pod = (i - self.edge_start_id) // (self.k // 2)
            for port_offset in range(self.k // 2):
                pc_index = pod * (self.k // 2) * (self.k // 2) + (i % (self.k // 2)) * (self.k // 2) + port_offset
                out_port = my_result[pc_index].append(trace_out[trace_out.cur_port == port_offset][self.used_cols],
                                                       ignore_index=True)
                my_result[pc_index] = out_port

    def trace_upd_multi_nodes(self, layer, node_list, my_result, my_link, model, device):
        all_trace_dict = dict()
        for node_id in node_list:
            trace_in = self.IN['DataFrame{}'.format(node_id)].copy()
            for c in range(self.total_switches):
                link_data = my_link[(c, node_id)]
                if link_data.shape[0] > 0:
                    trace_in = trace_in.append(link_data, ignore_index=True)
            all_trace_dict[node_id] = trace_in.sort_values('etime').reset_index(drop=True)

        trace_out_dict = self.infer_multi_nodes(all_trace_dict, layer, model, device)

        for node_id, trace_out in trace_out_dict.items():
            for connected_port, connected_node in enumerate(self.port_connection_matrix[node_id]):
                if connected_node > -1:
                    eport_flow = trace_out[trace_out.cur_port == connected_port]
                    if eport_flow.shape[0] > 0:
                        my_link[(node_id, connected_node)] = self.link_info_upd(eport_flow)
            if self.edge_start_id <= node_id <= self.edge_end_id:
                for port_offset in range(self.k // 2):
                    pc_index = (node_id - self.edge_start_id) * (self.k // 2) + port_offset
                    out_port = my_result[pc_index].append(trace_out[trace_out.cur_port == port_offset][self.used_cols],
                                                           ignore_index=True)
                    my_result[pc_index] = out_port

    def pod_infer_sync(self, gpu_number, device_idx, my_result, my_link, my_progress):
        print("Pod infer sync idx: ", device_idx)

        if self.model is None:
            model = self.load_model(return_model=True)
        else:
            model = copy.deepcopy(self.model)

        def distribute_load(total_items, num_gpus, gpu_id):
            items_per_gpu = total_items // num_gpus
            remainder = total_items % num_gpus
            start = gpu_id * items_per_gpu + min(gpu_id, remainder)
            end = start + items_per_gpu + (1 if gpu_id < remainder else 0)
            return start, end

        core_bg, core_ed = distribute_load(self.num_core, gpu_number, device_idx)
        core_bg += self.core_start_id
        core_ed += self.core_start_id

        agg_bg, agg_ed = distribute_load(self.num_agg, gpu_number, device_idx)
        agg_bg += self.agg_start_id
        agg_ed += self.agg_start_id

        edge_bg, edge_ed = distribute_load(self.num_edge, gpu_number, device_idx)
        edge_bg += self.edge_start_id
        edge_ed += self.edge_start_id

        pc_bg, pc_ed = distribute_load(self.num_hosts, gpu_number, device_idx)

        def sync(gpu_number, my_progress, idx):
            cur_progress = my_progress[idx]
            while True:
                wait = False
                for i in range(gpu_number):
                    if my_progress[i] < cur_progress:
                        wait = True
                        break
                if not wait:
                    break
                else:
                    time.sleep(0.01)

        device = torch.device(device_idx)
        model = model.to(device)
        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols)
        self.trace_upd_multi_nodes('edge', list(range(edge_bg, edge_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        for i in range(pc_bg, pc_ed): my_result[i] = pd.DataFrame(columns=self.used_cols)
        self.trace_upd_multi_nodes('agg', list(range(agg_bg, agg_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        self.trace_upd_multi_nodes('core', list(range(core_bg, core_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        self.trace_upd_multi_nodes('agg', list(range(agg_bg, agg_ed)), my_result, my_link, model, device)

        my_progress[device_idx] += 1
        sync(gpu_number, my_progress, device_idx)

        self.trace_upd_multi_nodes('edge', list(range(edge_bg, edge_ed)), my_result, my_link, model, device)

    def run_parallel(self, gpu_number=4):
        print("Run parallel")
        print("GPU number:", gpu_number)

        self.used_cols = ['index', 'timestamp (sec)', 'pkt len (byte)', 'priority', 'src_hub', 'src_port',
                          'cur_hub', 'cur_port', 'path', 'dep_time', 'etime']

        self.load_scaler()
        self.flows_to_port()

        LINKS = {(i, j): pd.DataFrame() for i in range(self.total_switches) for j in range(self.total_switches)}

        with Manager() as MG:
            my_result = MG.dict()
            my_link = MG.dict(LINKS)
            my_progress = MG.list([0 for _ in range(gpu_number)])

            t0 = time.time()
            threads = []
            for i in range(gpu_number):
                t = Process(target=self.pod_infer_sync, args=(gpu_number, i, my_result, my_link, my_progress))
                threads.append(t)
                t.start()
            for thr in threads:  thr.join()

            print("time used (total)", "%f min." % ((time.time() - t0) / 60))
            result = pd.DataFrame()
            for key in my_result.keys():
                result = pd.concat([result, my_result[key]], ignore_index=True)

            result.reset_index(drop=True).to_csv(self.output_file, index=False)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Fat-Tree network simulation.")

    parser.add_argument('--identifier', type=str, default="mse-5.14e-05", help='Identifier for the model.')
    parser.add_argument('--traffic_pattern', type=str, default="poisson", help='Traffic pattern to simulate.')
    parser.add_argument('--run_index', type=int, default=0, help='Index of the simulation run.')
    parser.add_argument('--k_val', type=int, default=8, help='K-value for the Fat-Tree topology.')
    parser.add_argument('--gpu_number', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--visible_gpus', nargs='+', type=int, default=[0, 1, 2, 3], help='Specify which GPU IDs to make visible.')

    args = parser.parse_args()

    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_gpus))

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"--- Verification Run: k={args.k_val}, traffic={args.traffic_pattern}, run_idx={args.run_index}, identifier={args.identifier} ---")
    ft = fattree(args.run_index, k=args.k_val, traffic_pattern=args.traffic_pattern, model_identifier=args.identifier)
    ft.run_parallel(gpu_number=args.gpu_number)
    print("--- Verification Run Complete ---")
