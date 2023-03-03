import torch
from time import time
import models
import utils
import dataloading
import losses

# task = "bandmaster_n4"
# task = "bandmaster_n6"
# task = "bandmaster_n8"
# task = "bandmaster_n10"
# task = "bandmaster_b4"
# task = "bandmaster_b6"
# task = "bandmaster_b8"
# task = "bandmaster_b10"
task = "jcm_c6"
# task = "jcm_b6"

if task == "bandmaster_n4":
    IN_FILE = 'wzmaki/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_4.wav'
    OUT_FILE = f'wzmaki/bandmaster_better/{FILE_NAME}'
    DELAY = 248
elif task == "bandmaster_b4":
    IN_FILE = 'wzmaki/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_4.wav'
    OUT_FILE = f'wzmaki/bandmaster_better/{FILE_NAME}'
    DELAY = 248
elif task == "jcm_c6":
    IN_FILE = 'wzmaki/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'testgit_jcm800_next clean gain 6.wav'
    OUT_FILE = f'wzmaki/jcm_800_damian_2/{FILE_NAME}'
    DELAY = 248
elif task == "jcm_b6":
    IN_FILE = 'wzmaki/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'testgit_jcm800_next boost gain 6.wav'
    OUT_FILE = f'wzmaki/jcm_800_damian_2/{FILE_NAME}'
    DELAY = 248

EPOCH_LEN = 100
LR = 0.000001 * 100

SAMPLES_AT_ONCE = 1
MLP_DEPTH = 5
PRED_SAMPLES = 8192 // 16
# MODEL = "RNN"
MODEL = "TCN"
# MODEL = "LSTM"
if MODEL == "RNN" or MODEL == "LSTM":
    LOOKBACK = PRED_SAMPLES * 2
elif MODEL == "TCN":
    LOOKBACK = PRED_SAMPLES + (2 ** (MLP_DEPTH + 1) - 1) * SAMPLES_AT_ONCE
HIDDEN_WIDTH = 8
# HIDDEN_WIDTH = 90
# HIDDEN_WIDTH = 64
# HIDDEN_WIDTH = 256
# HIDDEN_WIDTH = 16
BATCH_SIZE = 32

AUDIO_START = 1000
# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')

in_data, out_data = dataloading.load_data(IN_FILE, OUT_FILE, DELAY, AUDIO_START, DEVICE)

data_len = in_data.shape[0]

def sample_batch(batch_size, lookback, pred_samples):
    idx = torch.randint(lookback, data_len, (batch_size, 1,), device=DEVICE)
    idx = idx.repeat(1, lookback)
    idx_diff = torch.arange(lookback, device=DEVICE)
    idx = idx - idx_diff
    idx = idx.flip(1)
    in_idx = idx
    out_idx = idx[:, -pred_samples:]
    batch = (in_data[in_idx].to(DEVICE), out_data[out_idx].to(DEVICE))
    return batch

# module = models.MLPModel(lookback=LOOKBACK, depth=2, hidden_dim=HIDDEN_WIDTH, pred_samples=PRED_SAMPLES, activation=torch.tanh).to(DEVICE)
# module = models.LinearModel(LOOKBACK, PRED_SAMPLES).to(DEVICE)
if MODEL == "RNN":
    update_module = models.MLPModel(lookback=HIDDEN_WIDTH, depth=MLP_DEPTH, hidden_dim=HIDDEN_WIDTH, pred_samples=HIDDEN_WIDTH, activation=torch.tanh, act_last=True).to(DEVICE)
    module = models.RNNWrapper(obs_dim=SAMPLES_AT_ONCE, hidden_dim=HIDDEN_WIDTH, update_module=update_module, out_samples=PRED_SAMPLES)
elif MODEL == "TCN":
    module = models.TCN(SAMPLES_AT_ONCE, HIDDEN_WIDTH, MLP_DEPTH, activation=torch.tanh)
elif MODEL == "LSTM":
    module = models.LSTM(SAMPLES_AT_ONCE, HIDDEN_WIDTH, MLP_DEPTH, out_samples=PRED_SAMPLES)

model = models.NAMPModel(module).to(DEVICE)

module_name = module.__class__.__name__
if module_name == "RNNWrapper":
    model_name = f"{module_name}_{HIDDEN_WIDTH}_{MLP_DEPTH}_{SAMPLES_AT_ONCE}_{FILE_NAME}"
elif module_name == "TCN":
    model_name = f"{module_name}_{SAMPLES_AT_ONCE}_{HIDDEN_WIDTH}_{MLP_DEPTH}_{FILE_NAME}"
elif module_name == "LSTM":
    model_name = f"LSTM_{SAMPLES_AT_ONCE}_{HIDDEN_WIDTH}_{MLP_DEPTH}_{FILE_NAME}"
else:
    raise NotImplementedError("module name:", module_name)

checkpoint_name = f"./model_checkpoints/{model_name}"
optim = torch.optim.Adam(lr=LR, params=model.parameters())

utils.try_load_model(checkpoint_name, model, optim)

print(f"the model has {utils.count_params(model)} parameters")
    
for epoch in range(1000000):
    t1 = time()
    mses = []
    norm_mses = []
    maes = []
    norm_maes = []
    highest_error = 0
    sampling_time = 0
    t1 = time()
    print()
    for i in range(EPOCH_LEN):
        print(f"iteration {i+1}/{EPOCH_LEN}", end='\r')
        optim.zero_grad()
        ts1 = time()
        x, y = sample_batch(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_samples=PRED_SAMPLES)
        sampling_time += time() - ts1
        out = model(x) 
        loss =\
            losses.mse(out, y)\
            + losses.norm_mse(out, y)
        norm_mse = losses.norm_mse(out.detach(), y.detach()).item()
        norm_mae = losses.norm_mae(out.detach(), y.detach()).item()
        mse = losses.mse(out.detach(), y.detach()).item()
        mae = losses.mae(out.detach(), y.detach()).item()
        highest_error = max(highest_error, (y-out).abs().max().item())
        mses.append(mse)
        norm_mses.append(norm_mse)
        maes.append(mae)
        norm_maes.append(norm_mae)
        loss.backward()
        optim.step()
    utils.save_model(checkpoint_name, model, optim)
    json_weights = model.get_json_weights()
    utils.save_json_weights(json_weights, path=checkpoint_name)
    t2 = time()
    print()
    print()
    print(f"avg_it_time: {(t2 - t1) / EPOCH_LEN}")
    print("epoch norm mse:", sum(norm_mses) / EPOCH_LEN)
    print("epoch norm mae:", sum(norm_maes) / EPOCH_LEN)
    print("epoch mse:", sum(mses) / EPOCH_LEN)
    print("epoch mae:", sum(maes) / EPOCH_LEN)
    print("epoch highest error:", highest_error)
    print("time sampling:", sampling_time)
    print("total time:", t2 - t1)
