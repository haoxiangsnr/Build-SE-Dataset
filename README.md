# Build Speech Enhancement Dataset

Build speech enhancement dataset.

## 依赖

- tqdm
- pytorch
- librosa

## Usage

1. 在 data/clean 放置纯净语音，比如 [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)，在 data/noise 中放置噪声文件
2. 配置 config 文件，配置具体指定的参数
3. 运行 main.py，通过指定参数，使用不同的脚本合成时域的带噪语音数据集，或者频谱的带噪频谱数据集：

```python
合成带噪语音

optional arguments:
  -h, --help            show this help message and exit
  -C CONFIG, --config CONFIG
                        配置文件
  -S SCRIPT, --script SCRIPT
                        合成带噪语音所用的脚本文件名
  -O DIST, --dist DIST  输出目录
  -R RANDOM_SEED, --random_seed RANDOM_SEED
                        随机种子
```

例如：

```python
python main.py -C confing -S script
```

Todo

## Supported features

- time_domain: Speech level. The noisy waveform corresponds to the clean waveform.
- frequency_domain_0: Speech level. The noisy spectrum corresponds to the clean spectrum, and they are the same size.
- frequency_domain_1: Frame level. The noisy spectrum has multi-frames, and the clean speech is one frame. The center frame of the noisy spectrum is aligned with the frame of the clean speech.
- frequency_domain_2: Frame level. The noisy spectrum is multi-frames, and the clean speech is multi-frames. They are the same numbers of frames.


## ToDo

- [ ] Replace .npy for a more efficient format
- [ ] Add more param with extracting spectrum
