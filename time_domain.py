import json
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm
import argparse

from utils.utils import (add_noise_for_waveform,
                         corrected_the_length_of_noise_and_clean_speech,
                         load_wavs, prepare_empty_dirs)


def main(config, random_seed, dist):
    """构建时域上的语音增强数据集

    Steps:
        1. 加载纯净语音信号
        2. 加载噪声文件
        3. 在纯净语音信号上叠加噪声信号
        4. 分别存储带噪语音与纯净语音

    Arguments:
        config {dict} -- 配置信息
        random_seed {int} -- 随机种子
        dist {str} -- 输出结果的目录
    """
    np.random.seed(random_seed)
    dist_dir = Path(dist)

    for dataset_itx, dataset_cfg in enumerate(config["dataset"], start=1):
        dataset_dir = dist_dir / dataset_cfg["name"]
        prepare_empty_dirs([dataset_dir])

        print(f"============ Building set {dataset_itx}: {dataset_cfg['name']} set ============")
        # Clean Speech
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

        # Noise
        noise_cfg = dataset_cfg["noise"]
        noise_database_dir = Path(noise_cfg["database"])
        noise_ys = {}
        for noise_type in tqdm(noise_cfg["types"], desc="Loading noise files"):
            mixture, _ = librosa.load(
                (noise_database_dir / (noise_type + ".wav")).as_posix(), 
                sr=noise_cfg["sampling_rate"])
            noise_ys[noise_type] = mixture
        print("Loaded noise.")

        # Combin
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

                    mixture_store[basename_text] = mixture

            clean_store[num] = clean

        print(f"Synthesize finished，storing file...")
        np.save((dataset_dir / "clean.npy").as_posix(), clean_store)
        np.save((dataset_dir / "mixture.npy").as_posix(), mixture_store)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成带噪语音")
    parser.add_argument("-C", "--config", required=True, type=str, help="配置文件")
    parser.add_argument("-O", "--dist", default="./dist", type=str, help="输出目录")
    parser.add_argument("-S", "--random_seed", default=0, type=int, help="随机种子" )
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.random_seed, args.dist)