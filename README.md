# EpochColor

EpochColor colorizes black and white photos and film footage while keeping the original grain and detail exactly as they were. It's for one person working through an archive on a Linux desktop, with or without a GPU.

**This is milestone 2: a command line proof for photos and video.** No GUI yet, and video export is H.265 10-bit only. It answers the question everything else depends on: does the output look good when the model only predicts color and the original luma passes through untouched, and does that color hold still from frame to frame?

## How it works

The model never touches brightness. A black and white photo already holds all of it, so EpochColor predicts only the two color channels (Lab a/b) at a reduced size, scales them up snapped to the edges of the full-size image, and recombines them with the original L.

1. The source is reduced to luminance. A sepia or slightly tinted scan loses its tint here.
2. A copy at working size (512 px short side by default) is denoised. Only this copy goes to the model, since models read grain as texture and paint blotches into it.
3. The model predicts color.
4. Painted hints, if any, are enforced: the correction spreads through areas of similar brightness and stops at edges.
5. Color is scaled to full size with a guided filter, so it follows real edges and ignores grain.
6. Color meets the original luma. Out-of-gamut pixels lose chroma instead of having channels clipped, so brightness stays put. In testing the output L matches the source to within 0.001 L*.

## Prerequisites

- Python 3.10 or newer
- FFmpeg with libx265 for video. openSUSE's stock package leaves x265 out, so I use the Packman build. Check with `ffmpeg -encoders | grep x265`.
- PyTorch for the network models (`siggraph17`, `eccv16`). The `hints` model runs without it.
- About 300 MB of disk for both model weights

## Install

I run openSUSE with an AMD card, so in my case it's the ROCm wheel. It depends on your system.

1. `git clone https://github.com/NullAngst/EpochColor && cd EpochColor`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. Install the PyTorch wheel for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/). For me that's `pip install torch --index-url https://download.pytorch.org/whl/rocmX.Y`, with X.Y being whichever ROCm version the site lists right now. NVIDIA wants the CUDA index, Intel the XPU index, no GPU the CPU index.
4. `pip install -e ".[raw,heic]"`, since that pulls in camera RAW and HEIC support too. Drop either one if you don't need it.
5. `epochcolor device` to check what PyTorch found. ROCm shows up as a CUDA device, that's normal.
6. `epochcolor fetch all` to download the weights to `~/.local/share/epochcolor/models`. Set `EPOCHCOLOR_MODELS` to put them elsewhere.

## Use

Colorize one photo, automatic only:

```
epochcolor photo scan.tif
```

That writes `scan.color.png` next to the source, 16-bit if the source was more than 8-bit.

Colorize a whole folder into another one:

```
epochcolor photo ~/Pictures/family/*.tif --out-dir ~/Pictures/family-color --format tif
```

### Painting hints

The automatic pass guesses. It can't know a coat was navy. Paint a few strokes and it can.

1. Open the photo in GIMP, Krita, or whichever editor you prefer.
2. Add a new transparent layer on top.
3. Paint rough strokes in the right colors: a line across the coat, a dab on each eye, a stroke along the wall. You don't need to mask anything, the strokes spread to the object's edges on their own.
4. Hide the photo layer and export just the strokes as a PNG with transparency.
5. `epochcolor photo scan.tif --hints scan-hints.png`

A stroke in neutral grey keeps that area uncolored. You can also paint straight onto a copy of the photo and export that without transparency: anything with visible color counts as a hint. DON'T do that with a sepia or tinted scan, since the whole picture would count as painted. Use the transparent layer.

For a set, put hint files in a folder named after each photo (`scan01.png` for `scan01.tif`) and pass `--hints-dir`.

### Models

| Model | What it does |
| --- | --- |
| `siggraph17` (default) | Automatic, and reads your hints as input. The one to use. |
| `eccv16` | Automatic only, more muted. Hints still apply afterward. |
| `hints` | No network at all. Your strokes are the only color, everything unpainted stays grey. Good for full control, or for testing without PyTorch. |

Both network models are Zhang et al., BSD-2-Clause, from [richzhang/colorization](https://github.com/richzhang/colorization). They're from 2016 and 2017, so expect plausible and somewhat muted. Newer models come in later milestones through the same adapter.

### Video

```
epochcolor video reel.mkv --rf 18
```

That writes `reel.color.mkv`: H.265 10-bit at RF 18, BT.709 tags, every audio track copied over untouched.

The clip has to be a constant frame rate, and progressive. A variable frame rate clip gets refused with a message saying so, since every timing calculation would drift. Convert it in HandBrake first (Video tab, Constant Framerate), or whichever tool you prefer.

What happens to it:

1. Decode at working size, and split into shots wherever the picture jumps. Exposure flicker doesn't count as a jump, since each frame's brightness is evened out before comparing.
2. Denoise the model's copy over time: each frame is averaged with its neighbors after lining them up along optical flow. Grain changes every frame and the picture mostly doesn't, so this strips grain with far less smearing than a spatial filter.
3. Run the model on every frame.
4. Stabilize the color. Per-frame models drift in hue, the same wall going a little redder then a little greener. A running average carried along the motion, forward and backward through each shot, holds it steady. It forgets wherever the motion doesn't line up, so color never gets dragged onto something new, and it never crosses a cut.
5. Render at full size with the original luma and pipe it to x265.

Useful switches:

- `--frames 120` does only the first 120 frames. Do this first on a new reel.
- `--list-shots` prints the shots it found. If it missed a cut, lower `--shot-threshold` (default 6). If it chopped a fast pan, raise it.
- `--stabilize 0.9` is the default, an average of up to 10 frames each way. `0.95` holds color longer, `0` turns it off so you can see what the model does raw.
- `--tune-grain` sets x265's grain tuning, which keeps grain from turning to mush above RF 18 or so. Files get bigger.
- `--probe` checks the clip and lists the audio tracks without doing anything.

The model pass is the slow part, so its output is cached per shot in `~/.cache/epochcolor`. Run again with a different RF, grain, stabilizer or saturation setting and it skips straight to the render. Stop it partway with Ctrl+C and the finished shots stay cached, so the same command picks up where it left off. `epochcolor cache` shows the size, `epochcolor cache clear` empties it. Set `EPOCHCOLOR_CACHE` to move it somewhere with room, since a feature runs to several gigabytes.

To compare the two models on the same reel, just run it with `-m siggraph17` and then `-m eccv16`. Both stay cached.

### Options that matter

- `--grain 100` keeps the original luma, the default. Lower it to blend in the denoised copy, `0` for fully clean.
- `--denoise N` sets the strength of the model's denoise in L* units. Auto by default. Raise it if colored blotches show up in grainy areas.
- `--spread 0.15` is how far a hint travels through flat areas, as a fraction of the short side. Raise it when a stroke doesn't fill its object, lower it when it leaks.
- `--working-size 512` is the short side for color work. Higher costs time and memory, and rarely looks different, since soft color is the whole trick.
- `--saturation 1.2` is a plain chroma boost. Proper grading comes later.
- `--device cpu` forces CPU. `cuda:1` picks a second GPU.

## Known limits

Honest list, so nobody is surprised.

- **The network models haven't been run against their real weights yet.** My build environment couldn't reach the weight host. The architectures match the reference code and load test weights cleanly, and the weights loader errors out if anything doesn't fit. But SIGGRAPH17 hint handling uses the mask convention from the original interactive demo (ideepcolor), and that needs a check on real photos. If hints get ignored or come out wrong with `siggraph17` but work fine with `eccv16`, that's the first place to look: `MASK_CENT` in `epochcolor/models/zhang.py`.
- Edges come from brightness only. Two objects with the same grey value and no line between them will share a color fix. Add a stroke on the other object to hold it.
- The auto denoise strength assumes fine grain. Coarse, clumpy grain fools it into going too light. Set `--denoise` by hand.
- EXIF is kept for JPEG, WebP and 8-bit PNG. 16-bit PNG and TIFF get the sRGB profile but not the EXIF yet.
- The RAW develop path is written but hasn't been run against a real RAW file yet.
- Large scans are fine on memory, but a 100 MP file will take a while on the final recombine. That part is CPU-only for now.
- Video hints don't exist yet. Painting on a frame and having it carry through the shot is milestone 5. Video takes `siggraph17` or `eccv16`, not `hints`.
- Video export is x265 10-bit into MKV or MP4 only. The full codec, bitrate and hardware encoder matrix is milestone 3. Audio is copied as is, so a codec MP4 can't hold (FLAC in older players, PCM) needs MKV for now.
- Optical flow runs on the CPU (OpenCV DIS). It's not the bottleneck yet. The model is: on a 2-core CPU with no GPU, `siggraph17` ran at under one frame per second on 640x360. A GPU changes that completely.
- The grain slider below 100% uses a spatial denoise at full size for video, not the temporal one. The model's copy does get the temporal one.
- Shot detection catches hard cuts. Dissolves and fades may get split oddly or missed. Check with `--list-shots`.

## Tests

`pip install -e ".[test]" && pytest`. CI runs the same on CPU on every push.

## License

GPL-3.0-or-later. Model weights aren't part of this repo and keep their own licenses.

Now photos and reels go in black and white and come out in color with the grain they were shot with, the color holds still across each shot, and re-renders at a new quality setting cost minutes instead of hours.
