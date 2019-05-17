# Build Speech Enhancement Dataset

Build speech enhancement dataset.

## Dependencies

- tqdm
- pytorch
- librosa

## Supported features

- time_domain: Speech level. The noisy waveform corresponds to the clean waveform.
- time_domain_wav: Same as above, except that it will save the speech separately, instead of storing all the speech in the .pkl file.
- frequency_domain_0: Speech level. The noisy spectrum corresponds to the clean spectrum, and they are the same size.
- frequency_domain_1: Frame level. The noisy spectrum has multi-frames, and the clean speech is one frame. The center frame of the noisy spectrum is aligned with the frame of the clean speech.
- frequency_domain_2: Frame level. The noisy spectrum is multi-frames, and the clean speech is multi-frames. They are the same numbers of frames.
- mask_0: Frame level. The noisy spectrum has multi-frames, and the mask is one frame. The center frame of the noisy spectrum is aligned with the frame of the mask.

## 使用

```shell
python [time_domain.py| time_domain_wav.py |frequency_domain_0.py|frequency_domain_1.py|mask_0.py] -C config.json
```

## ToDo

- [x] Replace .npy for a more efficient format
- [ ] Add more param with extracting spectrum
- [ ] 添加 count 参数来配合 min_sampling
