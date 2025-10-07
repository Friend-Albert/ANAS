class RnnArgs:
    def __init__(self):
        self.n_input = 12
        self.n_emb = 140
        self.n_hidden = 140
        self.n_output = 1
        self.n_node = 8
        self.dropout_l = 0
        self.dropout_o = 0
        self.dropout_e = 0
        self.dropout_i = 0
        self.dropout_x = 0
        self.attention_head = 4
        self.attention_output = 64
        self.d_k = 16
        self.d_v = 16


class TrainArgs:
    def __init__(self):
        self.base_dir = "./data"
        self.cur_cell = "./logs/cell.json"
        self.model_path = "./model/"
        self.data_path = './logs/cell_data.json'

        self.isENAS = False
        self.use_sim = True
        self.enable_warmup = True

        self.cuda = True
        self.seed = 42
        self.bins = 100
        self.alpha = 1
        self.beta = 5e-5
        self.lr = 2e-3
        self.weight_decay = 4e-7
        self.bptt = 42

        self.collect_interval = 600
        self.warmup_interval = 400
        self.collect_sample = 120
        self.warmup_sample = 40


class ControllerArgs:
    def __init__(self):
        self.n_sample = 1
        self.training_step = 20000
        self.derive_step = 10
        self.shared_rnn_activations = ['sigmoid', 'tanh', 'relu', 'swish']
        self.num_blocks = 16
        self.n_encoder = 8
        self.n_decoder = 8

        self.controller_hid = 200
        self.reward_c = 1e-3
        self.lr = 0.00035
        self.softmax_t = 5.0
        self.tanh_c = 2.5
        self.baseline_decay = 0.95
        self.discount = 1.0
        self.entropy_coeff = 0.0001


class BaseConfig:
    test_size = 0.2  # train_test_split ratio
    modelname = '4-port switch/FIFO'
    sub_rt = 0.005  # subsampling for Eval.
    TIME_STEPS = 42
    BATCH_SIZE = 32 * 8
    no_of_port = 4
    no_of_buffer = 1
    ser_rate = 2.5 * 1024 ** 2
    sp_wgt = 0.
    seed = 0
    window = 63  # window size to cal. average service time.
    no_process = 4  # multi-processing:no of processes used.
    epochs = 6
    n_outputs = 1
    learning_rate = 0.001
    l2 = 0.1
    lstm_params = {'layer': 2, 'cell_neurons': [200, 100], 'keep_prob': 1}
    att = 64.  # attention output layer dim
    mul_head = 3
    mul_head_output_nodes = 32


class ModelConfig:
    scaler = './data/_scaler'
    model = './trained/model'
    # train_sample= './trained/sample/train.h5'
    # test1_sample= './trained/sample/test1.h5'
    # test2_sample= './trained/sample/test2.h5'
    bins = 100
    errorbins = './data/_error'
    error_correction = False


def print_all_members(obj):
    print(type(obj))
    all_members = dir(obj)
    for member in all_members:
        if not member.startswith('__') and not member.endswith('__'):
            value = getattr(obj, member)
            print(f"{member:<20}: {value}")


encoder_arg = RnnArgs()
decoder_arg = RnnArgs()
train_arg = TrainArgs()
ctl_arg = ControllerArgs()
