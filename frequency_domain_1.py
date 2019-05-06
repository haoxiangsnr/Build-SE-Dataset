import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from utils.extract_features import lps
from utils.utils import (add_noise_for_waveform,
                         corrected_the_length_of_noise_and_clean_speech,
                         load_wavs, prepare_empty_dirs, unfold_spectrum)


def main(config, random_seed, dist, n_pad):
    """
    构建*频域*上的语音增强数据集（Log Power Spectrum）
    每句带噪语音的时间步上都包含多帧，多帧的中心帧对应这个时间步上的一帧纯净语音
    中心帧前面的时间帧：
    中心帧后面的时间帧：
    TODO 文档等待进一步更新

    Steps:
        1. 加载纯净语音信号
        2. 加载噪声文件
        3. 在纯净语音信号上叠加噪声信号
        4. 分别计算 LPS 特征
        5. 将带噪语音的 LPS 特征进行拓展
        5. 分别存储带噪语音与纯净语音

    Args:
        n_pad (int): 带噪语音的拓展大小
        config (dict): 配置信息
        random_seed (int): 随机种子
        dist (str): 输出结果的目录

    Dataset:
        dataset_1/
            mixture.npy
            clean.npy
        ...

        mixture.npy is {
            "0001_babble_-5": (257 * 3 * , T),
            "0001_babble_-10": (257 * 3, T),
            ...
        }

        clean.npy is {
            "0001": (257, T),
            "0002": (257, T),
            ...
        }
    """
    global clean_lps
    np.random.seed(random_seed)
    dist_dir = Path(dist)

    # 以遍历的方式读取 config.json 中各个数据集的配置项
    for dataset_itx, dataset_cfg in enumerate(config["dataset"], start=1):
        dataset_dir = dist_dir / dataset_cfg["name"]
        prepare_empty_dirs([dataset_dir])
        print("=" * 12 + f"Building set {dataset_itx}: {dataset_cfg['name']} set" + "=" * 12)

        # 加载纯净语音信号，存至 list 中
        clean_cfg = dataset_cfg["clean"]
        clean_speech_paths = librosa.util.find_files(
            directory=clean_cfg["database"],
            ext=clean_cfg["ext"],
            recurse=clean_cfg["recurse"],
            limit=clean_cfg["limit"],
            offset=clean_cfg["offset"]
        )
        random.shuffle(clean_speech_paths)
        clean_ys = load_wavs(
            file_paths=clean_speech_paths,
            sr=clean_cfg["sampling_rate"],
            min_sampling=clean_cfg["min_sampling"],
        )
        print("Loaded clean speeches.")

        # 加载噪声信号，存至 dict 中
        noise_cfg = dataset_cfg["noise"]
        noise_database_dir = Path(noise_cfg["database"])
        noise_ys = {}
        for noise_type in tqdm(noise_cfg["types"], desc="Loading noise files"):
            mixture, _ = librosa.load(
                (noise_database_dir / (noise_type + ".wav")).as_posix(),
                sr=noise_cfg["sampling_rate"])
            noise_ys[noise_type] = mixture
        print("Loaded noise.")

        # 合成带噪语音
        mixture_store = {}
        clean_store = {}
        for i, clean in tqdm(enumerate(clean_ys, start=1), desc="合成带噪语音"):
            num = str(i).zfill(4)
            for snr in dataset_cfg["snr"]:
                for noise_type in noise_ys.keys():
                    basename_text = f"{num}_{noise_type}_{snr}"

                    clean, mixture = corrected_the_length_of_noise_and_clean_speech(
                        clean_y=clean,
                        noise_y=noise_ys[noise_type]
                    )

                    mixture = add_noise_for_waveform(clean, mixture, int(snr))
                    assert len(mixture) == len(clean) == len(mixture)

                    mixture_lps = lps(mixture)
                    clean_lps = lps(clean)
                    mixture_lps = unfold_spectrum(mixture_lps, n_pad=n_pad)

                    assert mixture_lps.shape[0] == clean_lps.shape[0] == 257
                    mixture_store[basename_text] = mixture_lps

            clean_store[num] = clean_lps

        print(f"Synthesize finished，storing file...")
        np.save((dataset_dir / "clean.npy").as_posix(), clean_store)
        np.save((dataset_dir / "mixture.npy").as_posix(), mixture_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成频域带噪语音（可拓展帧）")
    parser.add_argument("-C", "--config", required=True, type=str, help="配置文件")
    parser.add_argument("-S", "--random_seed", default=0, type=int, help="随机种子")
    parser.add_argument("-O", "--dist", default="./dist", type=str, help="输出目录")
    parser.add_argument("-P", "--n_pad", default=3, type=int, help="带噪语音需要拓展的大小")
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.random_seed, args.dist, args.n_pad)
