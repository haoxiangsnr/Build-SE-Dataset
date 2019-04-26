import argparse
import json
import importlib

import script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成带噪语音")
    parser.add_argument("-C", "--config", required=True, type=str, help="配置文件")
    parser.add_argument("-S", "--script", required=True, type=str, help="合成带噪语音所用脚本（波形文件数据集，LPS 特征，Mel频谱，...")
    args = parser.parse_args()

    config = json.load(open(args.config))
    script = importlib.import_module(f"script.{args.script}")
    script.main(config)
