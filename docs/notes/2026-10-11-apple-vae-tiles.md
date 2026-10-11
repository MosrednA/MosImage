# VAE tiles on small Macs, 11 October 2026

Status: Historical
Last verified: 2026-10-11
Current state: [Guide](../guide.md#macos-apple-silicon) and [roadmap](../roadmap.md).

**Recorded scope:** Follow-up to the [MLX comparison](../../benchmarks/MLX_KLEIN_2026-10-11.md), which found that diffusers' Flux2 VAE allocates 6.5 GiB of Metal memory for one 1024² decode and 1.3 GiB with 512 px tiles. Quality was measured on the RTX 5070 Ti, because the tiled arithmetic is the same on every device. The Mac memory figures come from that comparison.

## Decode only, not round trip

The comparison measured tiles on a full round trip (encode and decode both tiled), and it found 39.6–43.1 dB against untiled. Generated images are decoded only: their latents come from the transformer, not from a tiled encoder. Decoding the same untiled latents in tiles is less faithful, because the decoder normalizes each tile on its own statistics:

| Image | Untiled decode (VAE vs source) | 1024 px tiles | 768 px tiles | 512 px tiles | 512 px, overlap 0.5 |
|---|---|---|---|---|---|
| portrait 1024² | 2.38 GiB · 38.9 dB | untiled | 1.34 GiB · 28.3 dB | 0.60 GiB · 26.0 dB | 26.3 dB |
| garden 1216 × 832 | 2.29 GiB · 33.9 dB | 1.93 GiB · 39.2 dB | 1.34 GiB · 35.9 dB | 0.60 GiB · 31.0 dB | 31.5 dB |
| kitchen 1024² | 2.38 GiB · 45.5 dB | untiled | 1.34 GiB · 31.0 dB | 0.60 GiB · 27.7 dB | 28.3 dB |

Memory is the CUDA decode peak, and dB is PSNR against the untiled decode. The difference amplified eight times shows tile-shaped shifts in brightness and colour. In 1:1 crops the two decodes look alike. Twice the overlap barely helps. For scale, 4-bit weights move a Fast edit to 23.7–24.3 dB against BF16 in the comparison.

## Decision

Tiles only where memory requires them: on a Mac where Quality's peak plus 12 GB does not fit, the same rule that warms up Fast (`apple_vae_tiling`, default on below 36 GB with 4-bit weights). There, 4-bit Fast peaks at 15.6–16.1 GiB untiled, the whole of a 16 GB MacBook Air; 512 px tiles bring one decode from 6.5 to 1.3 GiB. Larger Macs keep the untiled decode. `MOSIMAGE_APPLE_VAE_TILING` overrides the default in either direction.

The editor now starts on the preset of the model the engine warms up, so a small Mac opens on Fast instead of Quality.
