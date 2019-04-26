"""
合成时域带噪语音
"""
import json
import random
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from utils.utils import add_noise_for_waveform, prepare_empty_dirs, load_wavs, \
    corrected_the_length_of_noise_and_clean_speech


def main(config):
    np.random.seed(config["random_seed"])

    dist_dir = Path(config["dist"])
    sr = config["sampling_rate"]

    for dataset_itx, cfg in enumerate(config["dataset"], start=1):
        print(f"============ Building set {dataset_itx}: {cfg['name']} set ============")
        dataset_dir: Path = dist_dir / cfg["name"]
        prepare_empty_dirs([dataset_dir])

        """# ============ clean speeches ============ #"""
        clean_cfg = cfg["clean"]
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
            sr=sr,
            min_sampling=clean_cfg["min_sampling"],
        )

        print("Loaded clean speeches.")

        """# ============ noise speeches ============ #"""
        noise_cfg = cfg["noise"]
        noise_database_dir = Path(noise_cfg["database"])
        noise_ys = {}
        for noise_type in tqdm(noise_cfg["types"], desc="Loading noise files"):
            mixture, _ = librosa.load((noise_database_dir / (noise_type + ".wav")).as_posix(), sr=sr)
            noise_ys[noise_type] = mixture
        print("Loaded noise.")

        """# ============ 合成带噪语音 ============ #"""
        mixture_store = {}
        clean_store = {}
        for i, clean in tqdm(enumerate(clean_ys, start=1), desc="合成带噪语音"):
            for snr in cfg["snr"]:
                for noise_type in noise_ys.keys():
                    num = str(i).zfill(4)
                    basename_text = f"{num}_{noise_type}_{snr}"

                    clean, mixture = corrected_the_length_of_noise_and_clean_speech(
                        clean_y=clean,
                        noise_y=noise_ys[noise_type]
                    )

                    mixture = add_noise_for_waveform(clean, mixture, int(snr))
                    assert len(mixture) == len(clean) == len(mixture)

                    mixture_store[basename_text] = mixture

                    if i == 1:
                        # 防止纯净语音重复存储
                        clean_store[num] = clean

        print(f"Synthesize finished，storing file...")
        np.save((dataset_dir / "clean.npy").as_posix(), clean_store)
        np.save((dataset_dir / "mixture.npy").as_posix(), mixture_store)

if __name__ == "__main__":
    config = json.load(open("../config/time_domain.json"))
    main(config)
