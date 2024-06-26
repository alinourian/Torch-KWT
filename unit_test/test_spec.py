import librosa
import torch
import torch.nn as nn
from utils.spec import melspectrogram, mfcc_torch, mel_filter, adaptive_mel_filter2


audio_settings = {
    "sr": 16000,
    "n_mels": 40,
    "n_fft": 480,
    "win_length": 480,
    "hop_length": 160,
    "center": False,
}


# def test_adaptive_filter():
#     '''
#         testing:
#             - adaptive_mel_filter
#     '''
#     sr = audio_settings["sr"]
#     device = audio_settings["device"]
#     n_mels = audio_settings["n_mels"]
#     n_fft = audio_settings["n_fft"]
#     bs = 512
#     fm = torch.rand(bs, 40, device=device, dtype=torch.float32, requires_grad=True)
#     bw = torch.rand(bs, 40, device=device, dtype=torch.float32, requires_grad=True)
#     mel_basis = adaptive_mel_filter2(bs, fm, bw, audio_settings, device=device)

#     path = 'yes.wav'
#     x = librosa.load(path, sr=sr)[0]
#     x = librosa.util.fix_length(x, size=sr)
#     x_spec_torch = torch.from_numpy(melspectrogram(y=x, **audio_settings)).float().unsqueeze(0)
#     x_mfcc_torch = mfcc_torch(S=x_spec_torch, n_mfcc=n_mels, n_fft=n_fft, mel_basis=mel_basis, device=device)

#     x_spec = librosa.feature.melspectrogram(y=x, **audio_settings)        
#     x_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(x_spec), n_mfcc=n_mels)
#     x_mfcc = torch.from_numpy(x_mfcc)

#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)


def test_librosa_torch():
    '''
        testing:
            - melspectrogram
            - mfcc_torch
            - mel_filter
    '''

    path = 'yes.wav'
    sr = audio_settings["sr"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_mels = audio_settings["n_mels"]
    n_fft = audio_settings["n_fft"]

    x = librosa.load(path, sr=sr)[0]
    x = librosa.util.fix_length(x, size=sr)

    x_spec = librosa.feature.melspectrogram(y=x, **audio_settings)        
    x_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(x_spec), n_mfcc=n_mels)
    x_mfcc = torch.from_numpy(x_mfcc)

    x_spec_torch = torch.from_numpy(melspectrogram(y=x, **audio_settings)).float().unsqueeze(0)
    x_mfcc_torch_default_filter = mfcc_torch(S=x_spec_torch, n_mfcc=n_mels, n_fft=n_fft, device=device)

    mel_basis = mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).to(device)
    x_mfcc_torch_custom_filter = mfcc_torch(S=x_spec_torch, n_mfcc=n_mels, n_fft=n_fft, mel_basis=mel_basis, device=device)

    err_1 = compare2tensors(x_mfcc, x_mfcc_torch_default_filter)
    err_2 = compare2tensors(x_mfcc, x_mfcc_torch_custom_filter)
    err_3 = compare2tensors(x_mfcc_torch_custom_filter, x_mfcc_torch_default_filter)
    print(f"x_mfcc, x_mfcc_torch_default_filter: {err_1}")
    print(f"x_mfcc, x_mfcc_torch_custom_filter: {err_2}")
    print(f"x_mfcc_torch_custom_filter, x_mfcc_torch_default_filter: {err_3}")
    assert err_1 < 1e-6
    assert err_2 < 1e-6
    assert err_3 < 1e-6
    print("test passed.")


def compare2tensors(a, b):
    return torch.sum((a - b) ** 2)