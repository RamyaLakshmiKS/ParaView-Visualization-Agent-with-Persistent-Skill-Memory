## Goal
- long-horizon: render every volume in data/ in one session

## Resolved facts
- bonsai: uint8 256x256x256, range [0,255]; bg cutoff ~15%, dense core opaque
- engine: uint8 256x256x128, range [0,255]; bg cutoff ~15%, dense core opaque
- mri_woman: uint16-le 256x256x109, range [2,4095]; bg cutoff ~15%, dense core opaque

## Current TF
- [mri_woman] range-relative TF: cutoff ~15%, ghost mid band, dense ramp >55%

## Open issues

## Discard log
- [bonsai] compressed 63 lines -> 1 fact
- [engine] compressed 63 lines -> 1 fact
- [mri_woman] compressed 63 lines -> 1 fact
