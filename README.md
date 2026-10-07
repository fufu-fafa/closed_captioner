# closed_captioner

Generates `.srt` subtitles from audio/video locally on the CPU. The audio is never sent to a third-party transcription service.

Transcription uses **Whisper medium** (OpenAI), run through **sherpa-onnx** as an ONNX model quantized to int8. The default transcription language is **Indonesian** (`id`).

## Pipeline

1. **Audio conversion**: `ffmpeg` converts the input to mono 16 kHz WAV.
2. **VAD segmentation**: Silero VAD splits the audio into speech segments. Subtitle timestamps come from these segments.
3. **Transcription**: each segment is transcribed with the language set to `id` (Indonesian) by default.
4. **SRT generation**: long sentences are split into subtitles of at most 2 lines, and each segment's duration is divided according to character count.

## Installation

Requires Python 3.10+ and `ffmpeg` (`brew install ffmpeg`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python captioner.py --download-models   # ~1 GB into ./models
```

## Usage

```bash
.venv/bin/python captioner.py recording.mp3            # -> recording.srt
.venv/bin/python captioner.py video.mp4 -o sub.srt
.venv/bin/python captioner.py audio.mp3 -l en --model small --max-chars 37
```

| Option | Default | Description |
|---|---|---|
| `-l, --language` | `id` | Whisper language code (Indonesian by default) |
| `--model` | `medium` | `tiny` / `base` / `small` / `medium` |
| `--max-chars` | `42` | Max characters per line |
| `--max-lines` | `2` | Max lines per subtitle |
| `--min-silence` | `0.5` | Pause (seconds) that separates VAD segments |
| `--max-speech` | `20` | Max segment length (Whisper is limited to 30 seconds) |
| `--threads` | CPU−1 | Threads used for inference |
| `-v, --verbose` | — | Print each segment's text above the progress bar |
