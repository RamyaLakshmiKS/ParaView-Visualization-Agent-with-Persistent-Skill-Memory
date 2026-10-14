## Goal
- long-horizon: render every volume in data/ in one session

## Resolved facts
- aneurism_256x256x256_uint8: uint8 256x256x256, range [0,255] (via filename); bg cutoff ~15%, dense core opaque
- backpack_512x512x373_uint16: uint16-le 512x512x373, range [0,4071] (via filename); bg cutoff ~15%, dense core opaque
- beechnut_1024x1024x1546_uint16: uint16-le 1024x1024x1546, range [0,65535] (via filename); bg cutoff ~15%, dense core opaque
- blunt_fin_256x128x64_uint8: uint8 256x128x64, range [0,253] (via filename); bg cutoff ~15%, dense core opaque
- bonsai: uint8 256x256x256, range [0,255] (via known-layout); bg cutoff ~15%, dense core opaque
- boston_teapot_256x256x178_uint8: uint8 256x256x178, range [0,255] (via filename); bg cutoff ~15%, dense core opaque
- bunny_512x512x361_uint16: uint16-le 512x512x361, range [0,63536] (via filename); bg cutoff ~15%, dense core opaque
- csafe_heptane_302x302x302_uint8: uint8 302x302x302, range [0,255] (via filename); bg cutoff ~15%, dense core opaque
- engine: uint8 256x256x128, range [0,255] (via known-layout); bg cutoff ~15%, dense core opaque
- mri_woman: uint16-le 256x256x109, range [2,4095] (via known-layout); bg cutoff ~15%, dense core opaque
- zeiss_680x680x680_uint8: uint8 680x680x680, range [0,224] (via filename); bg cutoff ~15%, dense core opaque

## Current TF
- [zeiss_680x680x680_uint8] range-relative TF: cutoff ~15%, ghost mid band, dense ramp >55%

## Open issues

## Discard log
- [aneurism_256x256x256_uint8] compressed 64 lines -> 1 fact
- [backpack_512x512x373_uint16] compressed 64 lines -> 1 fact
- [beechnut_1024x1024x1546_uint16] compressed 64 lines -> 1 fact
- [blunt_fin_256x128x64_uint8] compressed 64 lines -> 1 fact
- [bonsai] compressed 63 lines -> 1 fact
- [boston_teapot_256x256x178_uint8] compressed 64 lines -> 1 fact
- [bunny_512x512x361_uint16] compressed 64 lines -> 1 fact
- [csafe_heptane_302x302x302_uint8] compressed 62 lines -> 1 fact
- [engine] compressed 63 lines -> 1 fact
- [mri_woman] compressed 63 lines -> 1 fact
- [zeiss_680x680x680_uint8] compressed 62 lines -> 1 fact
