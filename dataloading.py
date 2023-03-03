import torchaudio

def load_data(in_file, out_file, delay, audio_start, device):
    in_data = torchaudio.load(in_file)[0][0].to(device)
    out_data = torchaudio.load(out_file)[0][0].to(device)
    out_data = out_data[delay:]
    in_data = in_data[audio_start:]
    out_data = out_data[audio_start:]
    min_len = min(in_data.shape[0], out_data.shape[0])
    in_data = in_data[:min_len]
    out_data = out_data[:min_len]

    return in_data, out_data