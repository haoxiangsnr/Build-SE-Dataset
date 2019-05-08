import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from utils.extract_features import mag
from utils.utils import (add_noise_for_waveform,
                         corrected_the_length_of_noise_and_clean_speech,
                         load_wavs, prepare_empty_dirs, input_normalization, unfold_spectrum)


def main(config, random_seed, dist, apply_normalization, n_pad):
    """
    构建 IRM（Ideal ratio mask）语音增强数据集
    数据集为语句级别，带噪语音和它相应纯净语音的频谱尺寸相同

    Steps:
        1. 加载纯净语音信号
        2. 加载噪声文件
        3. 在纯净语音信号上叠加噪声信号
        4. 计算频谱，mask等
        5. 分别存储带噪语音的频谱与 mask

    Args:
        config (dict): 配置信息
        random_seed (int): 随机种子
        dist (str): 输出结果的目录
        apply_normalization (bool): 是否对 mixture 语音进行规范化
        n_pad (int): mixture 语音中帧的拓展范围，拓展后中心帧对应 mask 中的一帧

    Dataset:
        dataset_1/
            mixture.npy
            mask.npy
        ...

        mixture.npy is {
            "0001_babble_-5": (257, T * (n_pad * 2 + 1)),
            "0001_babble_-10": (257, T * T * (n_pad * 2 + 1))
            ...
        }

        mask.npy is {
            "0001_babble_-5": (257, T),
            "0001_babble_-10": (257, T),
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
        mask_store = {}
        for i, clean in tqdm(enumerate(clean_ys, start=1), desc="合成带噪语音"):
            num = str(i).zfill(4)
            for snr in dataset_cfg["snr"]:
                for noise_type in noise_ys.keys():
                    basename_text = f"{num}_{noise_type}_{snr}"

                    clean, noise = corrected_the_length_of_noise_and_clean_speech(
                        clean_y=clean,
                        noise_y=noise_ys[noise_type]
                    )

                    mixture = add_noise_for_waveform(clean, noise, int(snr))

                    mixture_mag = mag(mixture)
                    clean_mag = mag(clean)
                    noise_mag = mag(noise)

                    if apply_normalization:
                        mixture_mag = input_normalization(mixture_mag)

                    mixture_mag = unfold_spectrum(mixture_mag, n_pad=n_pad)
                    mask = noise_mag / (noise_mag + clean_mag)

                    assert mixture_mag.shape[0] == mask.shape[0] == 257
                    mixture_store[basename_text] = mixture_mag
                    mask_store[basename_text] = mask

        print(f"Synthesize finished，storing file...")
        np.save((dataset_dir / "mask.npy").as_posix(), mask_store)
        np.save((dataset_dir / "mixture.npy").as_posix(), mixture_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成频域带噪语音")
    parser.add_argument("-C", "--config", required=True, type=str, help="配置文件")
    parser.add_argument("-S", "--random_seed", default=0, type=int, help="随机种子")
    parser.add_argument("-A", "--apply_normalization", action="store_true", help="对输入应用规范化，即减去均值除以标准差")
    parser.add_argument("-O", "--dist", default="./dist", type=str, help="输出目录")
    parser.add_argument("-P", "--n_pad", default=3, type=int, help="带噪语音需要拓展的大小")
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.random_seed, args.dist, args.apply_normalization, args.n_pad)
