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

def get_bandmaster_n4_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_4.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_n6_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_6.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_n8_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_8.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_n10_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster n_10.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_b4_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster b_4.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_b6_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster b_6.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_b8_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster b_8.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_bandmaster_b10_metadata(wzmaki_path):
    IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
    FILE_NAME = 'bandmaster b_10.wav'
    OUT_FILE = f'{wzmaki_path}/bandmaster_better/{FILE_NAME}'
    DELAY = 248

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY

def get_task_metadata(task, wzmaki_path):
    if task == "bandmaster_n4":
        return get_bandmaster_n4_metadata(wzmaki_path)
    elif task == "bandmaster_n6":
        return get_bandmaster_n6_metadata(wzmaki_path)
    elif task == "bandmaster_n8":
        return get_bandmaster_n8_metadata(wzmaki_path)
    elif task == "bandmaster_n10":
        return get_bandmaster_n10_metadata(wzmaki_path)
    elif task == "bandmaster_b4":
        return get_bandmaster_b4_metadata(wzmaki_path)
    elif task == "bandmaster_b6":
        return get_bandmaster_b6_metadata(wzmaki_path)
    elif task == "bandmaster_b8":
        return get_bandmaster_b8_metadata(wzmaki_path)
    elif task == "bandmaster_b10":
        return get_bandmaster_b10_metadata(wzmaki_path)
    elif task == "jcm_c6":
        IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
        FILE_NAME = 'testgit_jcm800_next clean gain 6.wav'
        OUT_FILE = f'{wzmaki_path}/jcm_800_damian_2/{FILE_NAME}'
        DELAY = 248
    elif task == "jcm_c10":
        IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
        FILE_NAME = 'testgit_jcm800_next clean gain 10.wav'
        OUT_FILE = f'{wzmaki_path}/jcm_800_damian_2/{FILE_NAME}'
        DELAY = 248
    elif task == "jcm_b6":
        IN_FILE = f'{wzmaki_path}/bandmaster_better/bandmaster audio_in.wav'
        FILE_NAME = 'testgit_jcm800_next boost gain 6.wav'
        OUT_FILE = f'{wzmaki_path}/jcm_800_damian_2/{FILE_NAME}'
        DELAY = 248
    else:
        raise ValueError(f"unknown task: {task}")

    return IN_FILE, FILE_NAME, OUT_FILE, DELAY