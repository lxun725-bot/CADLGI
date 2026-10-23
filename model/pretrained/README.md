# Pretrained backbones

Place these files here before using the published configurations:

| File | SHA256 |
|---|---|
| repvit_m1_0_distill_300e.pth | 283bb9865f705481785ac302f15b8bdd1a7b50556f6be8a5e576ec44422a0737 |
| swin_tiny_patch4_window7_224.pth | 9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3 |

RepViT can also obtain its upstream file through the existing `auto` loader. Swin requires the local file. These are upstream initialization weights, not trained CADLGI checkpoints. Weight binaries are ignored by Git and are not included in this code export. The SHA256 values identify the backbone initialization files.
