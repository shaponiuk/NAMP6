import numpy as np
import torch
import os
import json

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params

def try_load_model(checkpoint_path, model, optim):
    try:
        chpt = torch.load(checkpoint_path + "/chpt.pt")
        model.load_state_dict(chpt["model"])
        # optim.load_state_dict(chpt["optim"])
        print("loaded model weights")
    except Exception as e:
        print("failed to load the model checkpoint")
        print(e)

def save_model(checkpoint_path, model, optim):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }, checkpoint_path + "/chpt.pt")

def save_json_weights(json_weights, path):
    os.makedirs(path, exist_ok=True)
    json_weights_str = json.dumps(json_weights)
    json_weights_str = json_weights_str.replace("\"", "\\\"")
    json_file = open(f"{path}/weights.json", "w")
    json_file.write(json_weights_str)
    json_file.close()