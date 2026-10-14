## Goal
- volume render mri_woman.raw so dense tissue stands out and soft tissue is faint

## Resolved facts
- dtype/shape = uint16-LE 256x256x109 (corr 0.88 > 0.46)
- scalar range [2,4095], 12-bit; background below ~5%
- histogram: low-value air/noise dominates -> hard cutoff at ~5%

## Current TF
- cutoff 0; ghost soft band 0.08-0.34 (~1-8%); dense ramp >0.46 to 0.75-0.95
- color: cool blue-grey soft -> warm gold/white dense

## Open issues

## Discard log
- compressed 30 raw inspect lines -> 2 facts
- compressed 32 histogram bins -> 1 fact
- kept 1 issue from 3 render-log lines
- cleared 'Open issues' (1 lines) — resolved: skull+sutures read, soft tissue translucent
