import numpy as np
import torch
import scipy
import scipy.signal
import scipy.fftpack
import warnings

import librosa
from librosa import util
from librosa import filters
from torchaudio import functional as F

from librosa.util.exceptions import ParameterError
from librosa.core.spectrum import _spectrogram
from librosa.core.convert import fft_frequencies, mel_frequencies

from typing import Any, Optional, Union, Collection
from librosa._typing import _WindowSpec, _PadMode, _PadModeSTFT


def adaptive_mel_filter(bs, fm, audio_settings):
    sr: float = audio_settings["sr"]
    n_fft: int = audio_settings["n_fft"]
    n_mels: int = audio_settings["n_mels"]
    fmax = float(sr) / 2

    weights = torch.zeros((bs, n_mels, int(1 + n_fft // 2)), dtype=torch.float32)
    fftfreqs = torch.from_numpy(fft_frequencies(sr=sr, n_fft=n_fft))
    fm, _ = torch.sort(fmax * fm)
    mel_f = torch.cat((torch.zeros((bs, 1)), fm, fmax * torch.ones((bs, 1))), dim=1)
    print(mel_f.shape)
    
    for bs_iter in range(bs):
        fdiff = torch.diff(mel_f[bs_iter])
        ramps = mel_f[bs_iter].reshape(-1, 1) - fftfreqs
        for i in range(n_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[bs_iter, i] = torch.clamp(torch.minimum(lower, upper), min=0)

        enorm = 2.0 / (mel_f[bs_iter, 2 : n_mels + 2] - mel_f[bs_iter, :n_mels])
        weights[bs_iter] *= enorm[:, None]

    return weights


def mel_filter(
    *,
    sr: float = 16000,
    n_fft: int = 480,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax = None,
    ftype = 'linear',
) -> np.ndarray:

    if fmax is None:
        fmax = float(sr) / 2

    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=torch.float32)
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=False)

    fdiff = np.diff(mel_f)

    if ftype == 'linear':
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))
    else:
        for i in range(n_mels):
            weights[i]  = np.exp(-(fftfreqs - mel_f[i+1]) ** 2 / (2 * (fdiff[i] / 3) ** 2))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def power_to_db(
    S,
    ref = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
):
    S = torch.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")
    amin = torch.tensor(amin)

    if torch.is_complex(S):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(torch.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = torch.abs(S)
    else:
        magnitude = S

    ref_value = torch.abs(torch.tensor(ref))

    log_spec = 10.0 * torch.log10(torch.maximum(magnitude, amin))
    log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = torch.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def mfcc_torch(
    *,
    mel_basis = None,
    sr: float = 22050,
    S = None, 
    n_mfcc: int = 20,
    norm: Optional[str] = "ortho", 
    n_fft: int = 2048,
):
    if mel_basis is None:
        mel_basis = mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mfcc, ftype='linear')
        mel_basis = torch.from_numpy(mel_basis).to('cuda')

    melspec = torch.einsum("...ft,...mf->...mt", S, mel_basis)
    S = power_to_db(melspec)

    dct_mat = F.create_dct(n_mfcc, n_mfcc, norm).type(torch.float32).to('cuda')
    M = torch.einsum('...fm,ft->...tm', S, dct_mat)

    return M


def mfcc(
    *,
    mel_basis: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None, 
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho", 
    lifter: float = 0,
    n_fft: int = 2048,
    **kwargs: Any,
) -> np.ndarray:

    if mel_basis is None:
        mel_basis = filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mfcc, **kwargs)

    melspec: np.ndarray = np.einsum("...ft,...mf->...mt", S, mel_basis, optimize=True)
    S = librosa.power_to_db(melspec)

    M: np.ndarray = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[
        ..., :n_mfcc, :
    ]
    
    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
        LI = util.expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise Exception(f"MFCC lifter={lifter} must be a non-negative number")


def melspectrogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    return S