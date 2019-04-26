import os
import shutil

import librosa
import numpy as np
from tqdm import tqdm

def corrected_the_length_of_noise_and_clean_speech(clean_y, noise_y):
    """
    合成带噪语音前的长度矫正，使 len(clean_y) == len(noise_y)
    """
    if len(clean_y) < len(noise_y):
        # 大多数情况，噪声比语音长
        return clean_y, noise_y[:len(clean_y)]
    elif len(clean_y) > len(noise_y):
        # 极少数情况，噪声比纯净语音短。此时需要将噪声重复多次，直到可以达到纯净语音的长度
        pad_factor = (len(clean_y) // len(noise_y))  # 拓展系数为需要拓展的次数，不包括原有的
        padded_noise_y = noise_y
        for i in range(pad_factor):
            padded_noise_y = np.concatenate((padded_noise_y, noise_y))
        noise_y = padded_noise_y
        return clean_y, noise_y[:len(clean_y)]
    else:
        return clean_y, noise_y

def get_name_and_ext(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name, ext


def load_noises(noise_wav_paths):
    """
    根据噪声列表加载噪声
    Args:
        noise_wav_paths (list): 噪声文件的路径列表

    Returns:
        dict: {"babble": [signals]}
    """
    out = {}
    for noise_path in tqdm(noise_wav_paths, desc="Loading noises: "):
        name, _ = get_name_and_ext(noise_path)
        wav, _ = librosa.load(noise_path, sr=16000)
        out[name] = wav

    return out


def add_noise_for_waveform(s, n, db):
    """
    为语音文件叠加噪声
    ----
    para:
        s：原语音的时域信号
        n：噪声的时域信号
        db：信噪比
    ----
    return:
        叠加噪声后的语音
    """
    alpha = np.sqrt(
        np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
    )
    mix = s + alpha * n
    return mix


def prepare_empty_dirs(dirs: list):
    """
    建立空目录。若已经存在，则删除后创建。
    parents=True

    Args:
        dirs: Path list

    Returns:
        dirs 中各个目录的句柄
    """
    result = []
    for d in dirs:
        if d.exists():
            shutil.rmtree(d.as_posix())
        d.mkdir(parents=True, exist_ok=False)
        result.append(d)
    return result


def load_wavs(file_paths, sr=16000, min_sampling=0):
    """
    根据 file_paths 逐个加载 wav 文件

    可以指定：
    - wav 文件需要满足的最小采样点数
    - 需要加载的 wav 文件数量，直到遍历完整个 list 或 满足了 limit 指定的数量要求

    Args:
        file_paths: 候选集合，其中采样点数大于 minimum_sampling 的 wav 才能被加载成功
        limit: 要求加载的数量上限
        sr: 采样率
        min_sampling: 最小采样点数

    """
    wavs = []
    actual_num = 0

    for i, path in tqdm(enumerate(file_paths, start=1), desc="Loading wavs ..."):
        wav, _ = librosa.load(path, sr=sr)
        if len(wav) >= min_sampling:
            wavs.append(wav)
            actual_num += 1
        else:
            print(f"The length of {file_paths[i]} < min sampling ...")

    print(f"需加载 wav 文件数量为：{len(file_paths)}")
    print(f"实际加载 wav 文件数量为：{actual_num}")
    return wavs
