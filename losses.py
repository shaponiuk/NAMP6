import torch

def norm_mse(yp, yt):
    return ((yp - yt)**2 / (yt**2 + 0.00001)).mean()

def norm_mae(yp, yt):
    return ((yp - yt).abs() / (yt.abs() + 0.00001)).mean()

def mse(yp, yt):
    return ((yp - yt)**2).mean()

def mae(yp, yt):
    return torch.abs(yp - yt).mean()