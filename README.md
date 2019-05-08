# Build Speech Enhancement Dataset

Build speech enhancement dataset.

## Dependencies

- tqdm
- pytorch
- librosa

## Supported features

- time_domain: Speech level. The noisy waveform corresponds to the clean waveform.
- frequency_domain_0: Speech level. The noisy spectrum corresponds to the clean spectrum, and they are the same size.
- frequency_domain_1: Frame level. The noisy spectrum has multi-frames, and the clean speech is one frame. The center frame of the noisy spectrum is aligned with the frame of the clean speech.
- frequency_domain_2: Frame level. The noisy spectrum is multi-frames, and the clean speech is multi-frames. They are the same numbers of frames.
- mask_0: Frame level. The noisy spectrum has multi-frames, and the mask is one frame. The center frame of the noisy spectrum is aligned with the frame of the mask.

## ToDo

- [ ] Replace .npy for a more efficient format
- [ ] Add more param with extracting spectrum
