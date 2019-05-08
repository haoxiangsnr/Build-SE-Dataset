import librosa
import numpy as np


def lps(y, pad=0):
    """
    提取 Log Power Spectrum，仅支持 sr=16000 的波形

    Args:
        y: 信号
        pad: 左右填充

    Returns:
        lps: (257, T)
    """
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    lps = np.log(np.power(np.abs(D), 2))
    if pad != 0:
        lps = np.concatenate((np.zeros((257, pad)), lps, np.zeros((257, pad))), axis=1)
    return lps

def mag(y):
    D = librosa.stft(y, n_fft=512, hop_length=256, window='hamming')
    return np.abs(D)
