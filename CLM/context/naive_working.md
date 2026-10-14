## Goal
- volume render mri_woman.raw so dense tissue stands out and soft tissue is faint

## Resolved facts
- byte[0]=0x00
- byte[1]=0x25
- byte[2]=0x4a
- byte[3]=0x6f
- byte[4]=0x94
- byte[5]=0xb9
- byte[6]=0xde
- byte[7]=0x03
- byte[8]=0x28
- byte[9]=0x4d
- byte[10]=0x72
- byte[11]=0x97
- byte[12]=0xbc
- byte[13]=0xe1
- byte[14]=0x06
- byte[15]=0x2b
- byte[16]=0x50
- byte[17]=0x75
- byte[18]=0x9a
- byte[19]=0xbf
- byte[20]=0xe4
- byte[21]=0x09
- byte[22]=0x2e
- byte[23]=0x53
- file size = 14,286,848 bytes
- candidate A: uint8 256x256x218  (fits byte count)
- candidate B: uint16-LE 256x256x109 (fits byte count)
- adjacent-slice correlation A = 0.46
- adjacent-slice correlation B = 0.88
- scalar range if B (uint16-LE) = [2, 4095]  (looks 12-bit)
- bin[0]=96
- bin[1]=55
- bin[2]=14
- bin[3]=973
- bin[4]=932
- bin[5]=891
- bin[6]=850
- bin[7]=809
- bin[8]=768
- bin[9]=727
- bin[10]=686
- bin[11]=645
- bin[12]=604
- bin[13]=563
- bin[14]=522
- bin[15]=481
- bin[16]=440
- bin[17]=399
- bin[18]=358
- bin[19]=317
- bin[20]=276
- bin[21]=235
- bin[22]=194
- bin[23]=153
- bin[24]=112
- bin[25]=71
- bin[26]=30
- bin[27]=989
- bin[28]=948
- bin[29]=907
- bin[30]=866
- bin[31]=825

## Current TF

## Open issues
- wrote results/mri_woman.py (142 lines)
- vtk offscreen render -> results/mri_woman.png
- LOOK: skin shell too opaque, hides interior anatomy
- edited transfer function: ghost soft-tissue band 0.08-0.34
- vtk offscreen render -> results/mri_woman.png
- LOOK: skull + sutures read clearly, soft tissue translucent — good

## Discard log
