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
from typing import Any, Optional, Union, Collection
from librosa._typing import _WindowSpec, _PadMode, _PadModeSTFT


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
        mel_basis = filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mfcc).to('cuda')

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