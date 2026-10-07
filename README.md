# EpochColor

EpochColor colorizes black and white photos and film footage while keeping the original grain and detail exactly as they were. It's for one person working through an archive on a Linux desktop, with or without a GPU.

**Where it's at:** the editor with a before/after viewer, a timeline for multiple clips, painted colour hints that follow objects through a shot, saved colours, per-shot grading with keyframes and scopes, a model manager with newer models, and the full export matrix. Still to come: negative inversion, regrain and deflicker, the Flatpak, and Windows.

## Download and run

Grab `EpochColor-<version>-x86_64.AppImage` from the [Releases page](https://github.com/NullAngst/EpochColor/releases).

1. `chmod +x EpochColor-*-x86_64.AppImage`
2. `./EpochColor-*-x86_64.AppImage`, or double-click it.
3. On first start it offers to download PyTorch for your GPU: ROCm for AMD, CUDA for NVIDIA, XPU for Intel, CPU otherwise. Say yes, since the colorization models need it. It goes into `~/.local/share/epochcolor/torch-cpython-312`, separate from the AppImage, so app updates don't download it again. Colour > Install PyTorch brings this back later.
4. Colour > Model manager. Download a model: `siggraph17` (130 MB) is the default and reads painted hints, `ddcolor-tiny` (220 MB) is the newer one that still runs on a CPU, and the full DDColor models (912 MB) want a GPU.

That's it. FFmpeg (with x264, x265, SVT-AV1, NVENC, VAAPI, QSV and AMF) and Python are inside the AppImage, so nothing else needs installing. openSUSE's own FFmpeg doesn't matter here.

The same AppImage runs the command line, too: `./EpochColor-*-x86_64.AppImage video reel.mkv --rf 18`. Any of the commands below work that way. With no command, or with files, it opens the editor.

A few notes on that first-run PyTorch download:

- It checks PyTorch's index live and takes the newest build that fits: for NVIDIA, the newest CUDA build your driver can run, read from `/proc/driver/nvidia/version`. AMD needs `/dev/kfd`, which means the amdgpu kernel driver, standard on any current distro.
- Sizes: CPU about 200 MB, CUDA about 3 GB, ROCm 4 to 6 GB. While it unpacks it needs a lot more room than that, about 20 GB for ROCm and 12 GB for CUDA, and it checks before it starts.
- It unpacks in `~/.local/share/epochcolor/pip-tmp`, not `/tmp`. Why? Because openSUSE, Fedora and Arch mount `/tmp` as tmpfs, which lives in RAM and tops out at half of it, so a ROCm build fills it and fails with "no space left on device" while your disk sits there with hundreds of gigabytes free. Versions before 0.5.1 had exactly that bug. If `~/.local/share` itself is on a small partition, set `XDG_DATA_HOME` to somewhere bigger before starting.
- `epochcolor setup-torch --check` shows what it would pick. `--variant cpu` (or rocm, cuda, xpu) overrides it. `--remove` deletes it.
- A PyTorch you installed yourself always wins over the downloaded one.

The AppImage needs glibc 2.28 or newer and the usual desktop libraries (OpenGL/EGL, fontconfig, xkbcommon). Any current openSUSE, Fedora, Ubuntu or Arch has them.

## Disk, memory and the GPU

**Working folder.** Everything EpochColor makes while it works goes in one folder: proxies, the colour analysis, colorized photos, export scratch files. By default that's `~/.cache/epochcolor`. File > Working folder and limits moves it, or `epochcolor cache dir /mnt/big/epochcolor` from the command line (`epochcolor cache dir default` puts it back). I'd point it at your biggest disk, since the analysis keeps about 0.7 MB per frame, roughly 1 GB per minute of 24 fps footage, and a feature film runs past 100 GB. Before it starts on a clip it checks the room and says so if there isn't enough. Nothing gets moved when you switch: the old folder stays until you empty it, and proxies rebuild in the new one. `epochcolor cache` shows the size and the free space. `epochcolor cache clear` empties it, and only touches EpochColor's own subfolders, so pointing it at a whole disk is safe.

Versions before 0.5.2 wrote the whole clip at working size to `L.raw` first, then read all of it back into RAM. On a long clip that filled the disk and then the memory. That file is gone now: the model pass decodes the source again, shot by shot. Leftover `L.raw` and `L.npy` files from older versions get deleted the next time the worker starts or `epochcolor cache` runs.

**Memory and CPU.** The worker runs at nice 10 and leaves one core free, so the desktop stays usable. A watchdog stops it when it goes over its memory limit (75% of RAM by default) or when free memory drops under 6% of RAM (at most 2 GB), before the system starts swapping. You get a message saying which one, and finished shots stay cached. All three numbers are in File > Working folder and limits, or as `threads`, `memory_limit_gb` and `nice` in `~/.config/epochcolor/settings.json`. The command line has the same watchdog for `photo`, `video` and `render`.

**The GPU check.** The first time a model needs the GPU, each GPU gets a quick test in a throwaway process, checked against the CPU result. The first one that passes gets used, and if none does, the CPU does the work and you're told why. Why bother? Because a GPU build of PyTorch that can't drive a card often crashes outright instead of giving an error, and that crash used to take the worker down with "exit code -11" on every job. The usual AMD causes:

- An integrated GPU. Most Ryzen 7000 and newer desktop chips have one, ROCm lists it next to the real card, and then crashes on it. The check tries the card with the most memory first and pins it with `ROCR_VISIBLE_DEVICES`.
- A chip that isn't in the PyTorch build. An RX 6700 XT is gfx1031, for example, and needs `HSA_OVERRIDE_GFX_VERSION=10.3.0` to run the gfx1030 kernels. The check tries that when the card fails as is.

The result is saved in `~/.config/epochcolor/gpu-check.json` and reused until PyTorch, the GPUs, the kernel or EpochColor change. If you set any of those ROCm variables yourself, the check leaves them alone. Colour > Test the GPU again redoes it. `epochcolor device --test` does it in a terminal and prints every attempt, which is the thing to paste if your GPU still won't work.

**Logs.** The worker writes everything to `~/.local/state/epochcolor/logs/worker.log`, including output from the C libraries under it and a stack trace if it crashes. When the worker dies, the error says how (crash, out of memory, killed), and Show Details has the end of that log. The GPU check logs to `gpu-check.log` next to it.

Those `(null): No such file or directory` lines in the terminal come from the ROCm build of PyTorch. It bundles its own libdrm, which looks for a table of AMD GPU names relative to where it thinks Python is installed. That doesn't line up with how EpochColor installs PyTorch, so the lookup fails. It's harmless: the only effect is a generic GPU name. They go to the worker log now instead of your terminal.

## Run from source

For hacking on it, or if you'd rather not use an AppImage. Needs Python 3.10 or newer and an FFmpeg with libx264, libx265 and libsvtav1 in your PATH. openSUSE's stock package leaves x264 and x265 out, so I use the Packman build, it depends on your system. `epochcolor encoders` shows what yours has.

The quick way:

1. `git clone https://github.com/NullAngst/EpochColor && cd EpochColor`
2. `python3 epochcolor.py`

The first time, it asks to make a virtual environment in `.venv` next to the script and installs everything into it, so your system Python stays clean. After that, `python3 epochcolor.py` switches into `.venv` by itself and opens the editor. Subcommands work the same as the AppImage: `python3 epochcolor.py video reel.mkv --rf 18`. If the venv step fails, your distro split venv out into its own package: Debian and Ubuntu call it `python3-venv`, check yours.

PyTorch gets its own folder per Python version (`torch-cpython-313` and so on), so a source checkout on your distro's Python and the AppImage's 3.12 don't trip over each other.

The manual way, if you want the `epochcolor` command on your PATH inside the venv:

1. `git clone https://github.com/NullAngst/EpochColor && cd EpochColor`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[gui,raw,heic]"`, since that pulls in the editor, camera RAW and HEIC support. Drop any of them you don't need.
4. `epochcolor setup-torch` downloads PyTorch for your GPU, same as the AppImage. Or install it into the venv yourself from [pytorch.org](https://pytorch.org/get-started/locally/).
5. `epochcolor device` to check what PyTorch found. ROCm shows up as a CUDA device, that's normal.
6. `epochcolor models` lists what's available, `epochcolor fetch siggraph17` downloads one (the DDColor ones want `--accept-license` once you've read their model card). They go to `~/.local/share/epochcolor/models`; `epochcolor models dir /somewhere/else` moves them.
7. `epochcolor gui` for the editor.

## Making a release

The build workflow (`.github/workflows/build.yml`) runs the tests, builds the AppImage, checks it actually starts (CLI, a photo, the editor and its worker process, all headless), and attaches it to a GitHub release. It runs when a version tag is pushed:

```
git tag v0.4.0
git push origin v0.4.0
```

The version in the app comes from the tag, so there's nothing to bump by hand. To try a build without making a release, open the Actions tab, pick "build", and hit Run workflow; the AppImage shows up as a download at the bottom of the run's page.

To build one on your own machine, `bash packaging/appimage/build.sh`. It needs curl, python3 with venv, tar and xz, and puts the AppImage in `dist/`. The Python series and the FFmpeg release branch are the two variables at the top of the script; each picks up the newest patch release of its series on every build.

## How it works

The model never touches brightness. A black and white photo already holds all of it, so EpochColor predicts only the two color channels (Lab a/b) at a reduced size, scales them up snapped to the edges of the full-size image, and recombines them with the original L.

1. The source is reduced to luminance. A sepia or slightly tinted scan loses its tint here.
2. A copy at working size (512 px short side by default) is denoised. Only this copy goes to the model, since models read grain as texture and paint blotches into it.
3. The model predicts color.
4. Painted hints, if any, are enforced: the correction spreads through areas of similar brightness and stops at edges.
5. Color is scaled to full size with a guided filter, so it follows real edges and ignores grain.
6. Color meets the original luma. Out-of-gamut pixels lose chroma instead of having channels clipped, so brightness stays put. In testing the output L matches the source to within 0.001 L*.

## The editor

```
epochcolor gui
```

Or run the AppImage, or `epochcolor-gui`. `epochcolor gui reel1.mkv reel2.mkv` opens with clips already in, `epochcolor gui film.epochcolor` opens a saved project.

1. File > Add clips (Ctrl+I). Each clip gets probed, a small proxy built for scrubbing, and its shots detected, in the background. The first clip sets the project's frame rate and resolution, and a clip that doesn't match is refused, same rule as the command line.
2. Colour > Colorize all clips (Ctrl+Shift+R). This is the slow part. Progress shows in the Inspector, and Cancel stops it. Finished shots stay cached either way.
3. Scrub and play. The viewer shows the proxy while moving and swaps in a full-size frame from the real pipeline once the playhead rests, so what you judge when paused is what exports. 1, 2 and 3 switch between colour, before/after (drag the divider), and the original.
4. Edit: S splits at the playhead, Delete ripple-deletes the selected segment, drag a segment's edge to trim it, drag its middle onto another segment to move it, I and O set the export range, M drops a marker with a note. Click an audio track's name to switch it on or off.
5. File > Export video (Ctrl+E). Same options as the command line, with a live plan that says what the RF became and which audio tracks get re-encoded.

Everything is undoable, without limit, for the session: Ctrl+Z and Ctrl+Shift+Z. F1 lists every key. Projects save as `.epochcolor` JSON, which is plain enough to fix by hand if it ever breaks. Sources are never modified.

**Shots.** Yellow lines on the timeline are shot cuts. Color is carried within a shot and never across one, so a missed cut smears one scene's colors into the next. B adds or removes a cut at the playhead. Colorize that clip again afterward, and only the changed shots rerun.

**Photos.** File > Add photos switches the bottom panel to a filmstrip. Select one or more, Colour > Colorize selected photos, then File > Export photos to write PNG, 16-bit TIFF, JPEG or WebP into a folder.

**Painting hints.** The automatic pass guesses, and it can't know a coat was navy. Tell it.

1. Put the playhead on a frame where the object is clearly visible and press P (or Paint hints above the viewer).
2. Pick the colour with the swatch, or Ctrl+click somewhere in the picture to take a colour from it.
3. Drag a few strokes across the coat. You don't mask anything: the strokes spread out to the object's edges on their own. Grey paints "no colour here", for a white shirt or a grey wall. Right-click a stroke to remove it. [ and ] or the mouse wheel change the brush size.
4. A moment after each stroke the hint pass runs (switch "Apply as I paint" off to do it by hand with Ctrl+Return). The fix gets carried forward and backward through the whole shot along the motion, and fades where the object leaves the frame or something covers it.

Only the painted shot reruns, and the expensive model pass stays cached, so it takes seconds rather than minutes. Paint on several frames of a long shot and each frame of it takes the nearest fixes. White dots on the timeline mark painted frames.

Photos work the same way: paint in the viewer with the Photos tab open.

**Saved colours.** Paint a stroke, click Save colour, call it "Anna's coat, navy". In the Colours panel (next to Clips), Find in every shot and photo looks for that object everywhere else. By default the matches get marked in the viewer and listed, and you click Apply on the right ones. "Paint matches straight away" skips the asking, at the cost of sometimes painting the wrong thing. It's matching on what the model's own features say, at the moment you ask; nothing gets trained.

**Grading.** The Grade panel (a tab next to the Inspector) grades the shot under the playhead, or the photo you picked. In order of how it's applied:

- white balance: temperature and tint, or Pick neutral and click something that should be grey
- lift, gamma, gain and offset wheels, each with a master level under it
- contrast around a pivot, saturation, vibrance (pushes the dull colours more than the strong ones)
- curves: master, red, green, blue, plus hue vs saturation and hue vs hue
- a secondary: an HSL qualifier (Pick the colour to isolate, then Show the matte to see what it catches) and a shape mask, ellipse or rectangle, feathered, which Track through shot follows along the motion
- a 3D LUT (.cube), with a mix amount

Everything runs in 32-bit float and nothing clips until the export, so pushing gain up and pulling it back down loses nothing. The preview, the paused full-size frame, photo export and video export all use the same code.

A shot's grade is static until you press Add key. With two or more keys the values move between them, linear or eased, and an edit between keys adds a new one. Copy and paste grades between shots with Ctrl+Alt+C and Ctrl+Alt+V. Export grade as .cube writes the global part of a grade as a LUT for use elsewhere; a LUT maps colour to colour, so it can't carry the shape mask.

**Scopes.** The Scopes panel shows a waveform, RGB parade, vectorscope (with skin tone line and 75% targets) or histogram of what the viewer shows.

**Models.** Colour > Model manager lists the catalog: the two Zhang models and four DDColor variants. It downloads with progress and resume, checks every file's SHA-256 before keeping it, and asks you to accept a license before downloading anything whose terms aren't plainly open. Add from file takes your own weights for an architecture EpochColor knows (DDColor, or either Zhang model) plus a small JSON manifest; it test-loads them before keeping them. Storage folder moves everything somewhere with room. The catalog refreshes from this repo's `catalog.json`, so a new set of weights for a known architecture shows up without an app update.

**Settings.** The Inspector's Colour box holds the project settings. Model, working size and denoise change what the model sees, so they need a new colorize pass, and the clip shows as not colorized until it's done. Stabilize reruns only the quick stabilizer pass. Grain and saturation apply straight away. Colour > Model device picks the GPU, `auto` by default.

**From the command line.** `epochcolor render film.epochcolor -o film.mkv` exports a saved project, hints, grades and all, without the editor. It uses the export settings you last picked in the editor, or `--preset`.

**Under the hood.** Everything slow runs in a separate worker process, so a long render doesn't freeze the window and a GPU driver crash takes down only the worker. You get told, and the next job starts a fresh one.

## Use from the command line

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

| Model | Size | What it does |
| --- | --- | --- |
| `siggraph17` (default) | 130 MB | Zhang et al. 2017. Automatic, and reads your hints as input. Small and quick, muted colour. |
| `eccv16` | 125 MB | Zhang et al. 2016. Automatic only, more muted still. Hints still apply afterward. |
| `ddcolor-modelscope` | 912 MB | DDColor 2023, ConvNeXt-L. Far more saturated and specific. The authors' all-round pick. |
| `ddcolor-paper` | 912 MB | DDColor with the paper's weights, more conservative. |
| `ddcolor-artistic` | 912 MB | DDColor trained for bolder colour. Less faithful, more striking. |
| `ddcolor-tiny` | 220 MB | DDColor with ConvNeXt-T. Much faster, a little less accurate. The one to try on CPU. |
| `hints` | built in | No network at all. Your strokes are the only colour, everything unpainted stays grey. |

The Zhang models are BSD-2-Clause, from [richzhang/colorization](https://github.com/richzhang/colorization). DDColor's code is Apache-2.0 and ships inside EpochColor (in `epochcolor/models/ddcolor_arch`, with its license); its weights come from the authors' Hugging Face repos, and their terms are on those model cards.

`epochcolor models` lists them with what's installed. `epochcolor models add weights.pth manifest.json` adds your own, `epochcolor models remove ID` deletes one, `epochcolor models refresh` pulls the newest catalog.

### Video

```
epochcolor video reel.mkv --rf 18
```

That writes `reel.color.mkv`: H.265 10-bit at RF 18, BT.709 tags, every audio track copied over untouched. Export options are their own section below.

The clip has to be a constant frame rate, and progressive. A variable frame rate clip gets refused with a message saying so, since every timing calculation would drift. Convert it in HandBrake first (Video tab, Constant Framerate), or whichever tool you prefer.

What happens to it:

1. Decode at working size, and split into shots wherever the picture jumps. Exposure flicker doesn't count as a jump, since each frame's brightness is evened out before comparing.
2. Denoise the model's copy over time: each frame is averaged with its neighbors after lining them up along optical flow. Grain changes every frame and the picture mostly doesn't, so this strips grain with far less smearing than a spatial filter.
3. Run the model on every frame.
4. Stabilize the color. Per-frame models drift in hue, the same wall going a little redder then a little greener. A running average carried along the motion, forward and backward through each shot, holds it steady. It forgets wherever the motion doesn't line up, so color never gets dragged onto something new, and it never crosses a cut.
5. Render at full size with the original luma and pipe it to the encoder.

Useful switches:

- `--frames 120` does only the first 120 frames. Do this first on a new reel.
- `--list-shots` prints the shots it found. If it missed a cut, lower `--shot-threshold` (default 6). If it chopped a fast pan, raise it.
- `--stabilize 0.9` is the default, an average of up to 10 frames each way. `0.95` holds color longer, `0` turns it off so you can see what the model does raw.
- `--probe` checks the clip and lists the audio tracks without doing anything.

The model pass is the slow part, so its output is cached per shot in `~/.cache/epochcolor`. Run again with a different codec, RF, grain, stabilizer or saturation setting and it skips straight to the render. Stop it partway with Ctrl+C and the finished shots stay cached, so the same command picks up where it left off. `epochcolor cache` shows the size, `epochcolor cache clear` empties it. Set `EPOCHCOLOR_CACHE` to move it somewhere with room.

How much room? The cache keeps chroma at 256 px on the short side (the models' own output size, already softer than the chroma in 4:2:0 video), stored as 8-bit integers, plus the denoised luma at the same size for the stabilizer. That's about 0.7 MB per frame for a 16:9 source: roughly 1 GB per minute of 24 fps footage, so a 90 minute feature needs about 90 GB free while you work on it. The proxies the editor builds add a little on top. `epochcolor cache clear` frees all of it, and nothing in there is anything you can't rebuild.

The 256 px chroma is a trade. Where two objects meet at the same grey (a red barn against green grass can be exactly that), nothing in the luma marks the edge, so the colour boundary comes out a few pixels soft. `--chroma-size 512` (or Chroma size in the editor) keeps it crisper for about four times the disk.

To compare the two models on the same reel, just run it with `-m siggraph17` and then `-m eccv16`. Both stay cached.

### Export

Same model as HandBrake: a codec and bit depth, a container, then either a constant quality RF or a target bitrate. The container comes from the output name.

| Codec | Bit depth | Chroma | Containers | Encoders |
| --- | --- | --- | --- | --- |
| `h265` (default) | 8, 10 | 4:2:0, 4:2:2, 4:4:4 | MKV, MP4 | libx265, hevc_nvenc, hevc_vaapi, hevc_qsv, hevc_amf |
| `h264` | 8, 10 | 4:2:0, 4:2:2, 4:4:4 | MKV, MP4 | libx264, plus the same four hardware ones at 8-bit |
| `av1` | 8, 10 | 4:2:0 | MKV, MP4 | libsvtav1, av1_nvenc, av1_vaapi, av1_qsv, av1_amf |
| `prores` | 10 | 4:2:2, 4:4:4 | MOV, MKV | prores_ks |
| `ffv1` | 8, 10, 12, 16 | all three | MKV | ffv1 |

Hardware encoders do 4:2:0 only. 4:2:2 and 4:4:4 go to software.

Some examples:

```
epochcolor video reel.mkv --rf 16 --speed slow --tune-grain
epochcolor video reel.mkv -o reel.mp4 --codec h264 --bits 8 --rf 20
epochcolor video reel.mkv --codec av1 --rf 30 --film-grain 8
epochcolor video reel.mkv -o reel.mov --codec prores
epochcolor video reel.mkv --bitrate 8000 --two-pass
epochcolor video reel.mkv --rf 0
```

**RF.** 0 to 51, lower is better and bigger. Each encoder takes it in its own mode: CRF on x264, x265 and SVT-AV1, CQ on NVENC, ICQ on QSV, QP on VAAPI and AMF. The plan printed before the render says which one it became, since the same number gives different sizes on GPU and CPU. SVT-AV1 goes up to 63.

**RF 0** is real lossless wherever the encoder has it: x264 QP 0, x265's lossless flag, NVENC's lossless tune. The tests check that x264 and x265 at RF 0 decode bit-for-bit identical to an FFV1 encode of the same frames. VAAPI, QSV, AMF, AV1 NVENC and SVT-AV1 have no lossless mode, so RF 0 there drops to the lowest QP with a warning that it is NOT lossless. ProRes ignores RF since its quality is set by the profile, and FFV1 is always lossless.

**Bitrate** is `--bitrate` in kbps. `--two-pass` works on x264 and x265. The frames get rendered once into a lossless FFV1 file in the cache, then both passes read it, since rendering is the expensive part. That file is as big as the export is long and gets deleted afterward.

**Encoder.** `--encoder auto` is the default. It takes the first hardware encoder that passes a test encode, in the order NVENC, VAAPI, QSV, AMF, and falls back to software. It also stays on software when the job needs something the hardware can't do: RF 0 lossless, 4:2:2 or 4:4:4, grain tuning, two-pass, or a speed name only the software encoder knows (`--speed slow` means x265). `--encoder software` or `--encoder hardware` forces either side, `--encoder hevc_vaapi` names one.

`epochcolor encoders` runs a two-frame test encode on every hardware encoder and prints what works. An encoder showing up in `ffmpeg -encoders` only means it was compiled in, not that the driver behind it works. Results are cached per ffmpeg binary. `--retest` after a driver update. On my AMD card under Linux that's VAAPI; AMF is mostly a Windows thing.

**Speed** is `--speed`, in the encoder's own names: `ultrafast` to `placebo` for x264 and x265, `p1` to `p7` for NVENC, `0` to `13` for SVT-AV1 (lower is slower and better), `veryfast` to `veryslow` for QSV, `speed`, `balanced` or `quality` for AMF. VAAPI has none.

**Grain.** Grain is the hardest thing to compress. `--tune-grain` keeps x264 and x265 from smearing it at higher RF, for bigger files. `--film-grain 8` on SVT-AV1 strips the grain before encoding and has the player regenerate it, so files get far smaller. But the grain you see then is synthetic, no longer the grain that was filmed.

**Audio.** Every track goes in by default, in order, each as its own track. `--audio-tracks 1,3` picks some, `--audio-tracks none` or `--no-audio` drops them. Tracks are copied as they are wherever the container takes them: MKV takes anything, MP4 takes AAC, MP3, AC3, E-AC3, Opus, ALAC and FLAC, MOV takes AAC, ALAC, PCM and a few others. A track that doesn't fit has to be re-encoded, and you get asked which codec: AAC, Opus or FLAC, filtered to what the container allows. Pass `--audio-fallback aac` (or whichever) to answer up front in scripts, or `--audio-codec` to re-encode every track. `--audio-bitrate` sets AAC or Opus kbps per track, otherwise it's 96 per channel for AAC and 64 per channel for Opus.

**Presets.** `epochcolor presets` lists them. Five are built in:

- `archive-h265`: H.265 10-bit RF 16, slow, grain tuned
- `share-h264`: H.264 8-bit RF 20, MP4, AAC
- `small-av1`: SVT-AV1 10-bit RF 30, Opus
- `edit-prores`: ProRes 422 HQ, MOV
- `master-ffv1`: FFV1 10-bit 4:4:4, lossless

`--preset archive-h265` loads one, and any flag after it overrides that one setting. `--save-preset mine` saves the current export settings as JSON in `~/.config/epochcolor/presets/mine.json`, which you can edit by hand, tinker as you see fit. A path to a `.json` file works as a preset too. `--dry-run` prints the plan without rendering anything, which is the quick way to check a preset.

Why 10-bit by default? Colorized footage is mostly smooth gradients: skies, skin, painted walls. 8-bit bands on those. 10-bit H.265 at the same RF holds them clean for a small size cost.

### Options that matter

- `--grain 100` keeps the original luma, the default. Lower it to blend in the denoised copy, `0` for fully clean.
- `--denoise N` sets the strength of the model's denoise in L* units. Auto by default. Raise it if colored blotches show up in grainy areas.
- `--spread 0.15` is how far a hint travels through flat areas, as a fraction of the short side. Raise it when a stroke doesn't fill its object, lower it when it leaks.
- `--working-size 512` is the short side for color work. Higher costs time and memory, and rarely looks different, since soft color is the whole trick.
- `--saturation 1.2` is a plain chroma boost. For real grading, use the editor's Grade panel and `epochcolor render`.
- `--device cpu` forces CPU. `cuda:1` picks a second GPU.

## Known limits

Honest list, so nobody is surprised.

- **The network models haven't been run against their real weights yet.** My build environment couldn't reach the weight host. The architectures match the reference code and load test weights cleanly, and the weights loader errors out if anything doesn't fit. But SIGGRAPH17 hint handling uses the mask convention from the original interactive demo (ideepcolor), and that needs a check on real photos. If hints get ignored or come out wrong with `siggraph17` but work fine with `eccv16`, that's the first place to look: `MASK_CENT` in `epochcolor/models/zhang.py`.
- Edges come from brightness only. Two objects with the same grey value and no line between them will share a color fix. Add a stroke on the other object to hold it.
- The auto denoise strength assumes fine grain. Coarse, clumpy grain fools it into going too light. Set `--denoise` by hand.
- EXIF is kept for JPEG, WebP and 8-bit PNG. 16-bit PNG and TIFF get the sRGB profile but not the EXIF yet.
- The RAW develop path is written but hasn't been run against a real RAW file yet.
- Large scans are fine on memory, but a 100 MP file will take a while on the final recombine. That part is CPU-only for now.
- Video needs a network model; the `hints` model is for photos only.
- **DDColor has never run on its real weights.** The architecture is the authors' own code, vendored unchanged, and builds with the paper's parameter counts (55.0M tiny, 227.9M large), but my build machine couldn't reach Hugging Face. The loader refuses weights that don't fit exactly, so a mismatch fails loudly instead of producing garbage. Which file each Hugging Face repo holds is looked up at download time, and its SHA-256 comes from Hugging Face's own listing.
- **DDColor's weight licenses are marked unverified.** The code is Apache-2.0, but I couldn't read the model cards from here, so the manager asks you to accept the terms before downloading. Read the card.
- Hint carrying is translation along optical flow plus an edge-aware fill. It holds well on things that move and turn slowly, and fades on fast motion, heavy motion blur, or objects that leave and come back. Paint another frame where it fades.
- Saved-colour matching uses the model's features as they are. It finds the same object in other shots well enough to be useful and also finds look-alikes; that's why it asks by default. The `hints` model has no features, so matching needs a network model.
- The qualifier and secondary corrections work in HSV, which is quick but shifts brightness a little when saturation changes a lot. Mask tracking follows position only, not scale or rotation.
- A LUT is stored as a path in the project. Move the .cube and the grade loses it (you're told when it can't be read).
- **None of the hardware encoders have run on real hardware yet.** My build machine has no GPU. Their arguments follow FFmpeg's documentation, `epochcolor encoders` proves whether each one starts, and auto falls back to software when a test encode fails. But the quality mappings (CQ, ICQ, QP) haven't been compared against the software encoders on real footage. VAAPI on AMD is what I'll check first, since that's my hardware. NVENC, QSV and AMF reports are welcome.
- ProRes 4444 is stored as 12-bit inside whatever goes in, that's how the format works. The source is 10-bit anyway.
- FLAC in MP4 is legal and FFmpeg writes it, but some players still skip the track. Use MKV, or re-encode to AAC, if that matters.
- Audio passthrough with `--frames` trims at the nearest audio packet, not the exact frame. Fine for a test render.
- An edited timeline (any trim, split, cut, join or in/out range) re-encodes every audio track, since compressed audio frames don't line up with video frames. The export dialog says so and asks for the codec. Only a single untouched clip passes audio through as is. Each piece of audio gets padded or cut to its exact video length, so a source whose audio runs short can't pull later pieces out of sync.
- **Playback in the editor is silent.** Audio shows as waveforms and goes into the export, but the preview doesn't play it yet.
- The working folder takes about 1 GB per minute of footage, so a feature needs real disk space while you work on it. Point it at a big disk (see Disk, memory and the GPU). Compressing it per shot is the obvious next step.
- The two-pass export writes a lossless FFV1 intermediate of the whole timeline into the working folder first. At 4K that's very large. The free-space check doesn't cover it yet, so leave room or use single pass.
- The GPU check hasn't run on real AMD hardware. My build machine has no GPU. The test it runs and the CPU fallback are tested, but which `ROCR_VISIBLE_DEVICES` index lands on which card comes from ROCm's own numbering, which I couldn't check. `epochcolor device --test` prints the name of each card it tries, so a mix-up would show there.
- The memory watchdog reads `/proc`, so it only works on Linux.
- Linux only for now. The Flatpak, its repo on GitHub Pages, and Windows builds are still milestone 8 and 9 work.
- The first-run PyTorch download takes the newest build PyTorch offers for your hardware at that moment. The spec's pinned torch stack with a weekly test-and-bump workflow isn't built yet, so if a brand new PyTorch release breaks something, `epochcolor setup-torch --variant <yours>` after a fix, or report it.
- The editor draws through OpenGL where it can and falls back to plain painting where it can't (some VMs, remote sessions). My build machine had no GPU, so the GL path is untested. The fallback is what the tests ran on.
- Optical flow runs on the CPU (OpenCV DIS). It's not the bottleneck yet. The model is: on a 2-core CPU with no GPU, `siggraph17` ran at under one frame per second on 640x360. A GPU changes that completely.
- The grain slider below 100% uses a spatial denoise at full size for video, not the temporal one. The model's copy does get the temporal one.
- Shot detection catches hard cuts. Dissolves and fades may get split oddly or missed. Check with `--list-shots`.

## Tests

`pip install -e ".[test,gui]" && pytest`. The editor tests run headless (`QT_QPA_PLATFORM=offscreen`), drive the real worker process, and export through it. CI runs the same on CPU on every push.

## License

GPL-3.0-or-later. Model weights aren't part of this repo and keep their own licenses.

Now reels and photos go into an editor in black and white, get cut, trimmed and checked shot by shot against the original, and come out in color with the grain they were shot with, in whichever codec, container and quality you pick, with every audio track where it should be.
