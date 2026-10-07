#!/usr/bin/env python3
"""Generate SRT subtitles from an audio/video file, locally.

Pipeline:
  1. ffmpeg      -> mono 16 kHz WAV
  2. Silero VAD  -> speech segments (source of the timestamps)
  3. Whisper     -> transcribe each segment (sherpa-onnx, int8, CPU)
  4. SRT         -> max 2 lines per subtitle, duration split by character count
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"

HF_BASE = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-{size}/resolve/main"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple progress bar on stderr: percent, count, elapsed time and ETA.

    If stderr is not a terminal (e.g. redirected to a file), nothing is drawn.
    """

    def __init__(self, total: float, label: str = "", unit: str = "", fmt=None):
        self.total = max(total, 1e-9)
        self.label = label
        self.unit = unit
        self.fmt = fmt or (lambda v: f"{v:.0f}")
        self.value = 0.0
        self.start = time.time()
        self.last_draw = 0.0
        self.enabled = sys.stderr.isatty()

    def update(self, value: float, force: bool = False) -> None:
        self.value = min(value, self.total)
        now = time.time()
        if force or now - self.last_draw >= 0.1 or self.value >= self.total:
            self.last_draw = now
            self.draw()

    def advance(self, amount: float) -> None:
        self.update(self.value + amount)

    def draw(self) -> None:
        if not self.enabled:
            return
        frac = self.value / self.total
        elapsed = time.time() - self.start
        eta = elapsed / frac - elapsed if frac > 0 else 0
        info = (f" {frac * 100:5.1f}%  {self.fmt(self.value)}/{self.fmt(self.total)}{self.unit}"
                f"  {fmt_clock(elapsed)}<{fmt_clock(eta)}")
        cols = shutil.get_terminal_size((80, 20)).columns
        width = max(10, min(40, cols - len(self.label) - len(info) - 3))
        filled = frac * width
        full = int(filled)
        partial = " ▏▎▍▌▋▊▉"[int((filled - full) * 8)] if full < width else ""
        bar = ("█" * full + partial).ljust(width)
        sys.stderr.write(f"\r\033[K{self.label}|{bar}|{info}")
        sys.stderr.flush()

    def write(self, text: str) -> None:
        """Print a line above the bar without breaking it."""
        if self.enabled:
            sys.stderr.write("\r\033[K")
        print(text, file=sys.stderr)
        self.draw()

    def close(self) -> None:
        self.update(self.total, force=True)
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def fmt_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_mb(n: float) -> str:
    return f"{n / 1e6:.0f}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_files(model_dir: Path, size: str) -> dict:
    return {
        "encoder": model_dir / f"{size}-encoder.int8.onnx",
        "decoder": model_dir / f"{size}-decoder.int8.onnx",
        "tokens": model_dir / f"{size}-tokens.txt",
        "vad": model_dir / "silero_vad.onnx",
    }


def download_models(model_dir: Path, size: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    base = HF_BASE.format(size=size)
    for key, path in model_files(model_dir, size).items():
        if path.exists() and path.stat().st_size > 0:
            continue
        url = VAD_URL if key == "vad" else f"{base}/{path.name}"
        print(f"Downloading {path.name} ...", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        for attempt in range(10):
            try:
                fetch_resumable(url, tmp)
                break
            except OSError as e:
                print(f"  connection lost ({e}), retrying ...", file=sys.stderr)
                time.sleep(2)
        else:
            sys.exit(f"Failed to download {url}")
        tmp.rename(path)


def fetch_resumable(url: str, dst: Path) -> None:
    """Download `url` to `dst`, resuming from the size of any existing file."""
    offset = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        resumed = offset and resp.status == 206
        if not resumed:
            offset = 0
        total = offset + int(resp.headers.get("Content-Length") or 0)
        bar = ProgressBar(total, f"  {dst.stem} ", " MB", fmt_mb)
        bar.update(offset)
        with open(dst, "ab" if resumed else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.advance(len(chunk))
        bar.close()


# ---------------------------------------------------------------------------
# 1. Audio conversion
# ---------------------------------------------------------------------------

def convert_to_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1 and f.getframerate() == SAMPLE_RATE
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# 2. VAD segmentation
# ---------------------------------------------------------------------------

def vad_segments(samples: np.ndarray, vad_model: Path, args) -> list[tuple[float, np.ndarray]]:
    """Return a list of (start_seconds, samples) for each speech segment."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence
    config.silero_vad.min_speech_duration = 0.25
    # Whisper only accepts up to 30 seconds per input.
    config.silero_vad.max_speech_duration = args.max_speech
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1

    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)

    segments = []

    def drain():
        while not vad.empty():
            seg = vad.front
            segments.append((seg.start / SAMPLE_RATE, np.array(seg.samples, dtype=np.float32)))
            vad.pop()

    bar = ProgressBar(len(samples) / SAMPLE_RATE, "      VAD ", " s", fmt_clock)
    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        drain()
        bar.update(i / SAMPLE_RATE)
    vad.flush()
    drain()
    bar.close()
    return segments


# ---------------------------------------------------------------------------
# 3. Transcription
# ---------------------------------------------------------------------------

def create_recognizer(files: dict, args) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        tokens=str(files["tokens"]),
        language=args.language,
        task="transcribe",
        num_threads=args.threads,
        tail_paddings=-1,
    )


def transcribe(recognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


# ---------------------------------------------------------------------------
# 4. SRT generation
# ---------------------------------------------------------------------------

def wrap_words(words: list[str], max_chars: int) -> list[str]:
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def split_into_cues(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Split text into cues of at most `max_lines` lines x `max_chars` characters.

    Sentence boundaries take priority; sentences that are too long are word-wrapped
    and spread evenly over as few cues as possible.
    """
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s]
    cues = []
    for sentence in sentences:
        words = sentence.split()
        lines = wrap_words(words, max_chars)
        n_cues = -(-len(lines) // max_lines)
        balanced = [wrap_words(chunk, max_chars) for chunk in split_evenly(words, n_cues)]
        if all(len(c) <= max_lines for c in balanced):
            cues.extend("\n".join(c) for c in balanced)
        else:
            for i in range(0, len(lines), max_lines):
                cues.append("\n".join(lines[i:i + max_lines]))
    return cues


def split_evenly(words: list[str], n: int) -> list[list[str]]:
    """Split words into `n` groups with roughly equal character counts."""
    if n <= 1:
        return [words]
    total = sum(len(w) + 1 for w in words)
    chunks, cur, acc = [], [], 0
    for w in words:
        cur.append(w)
        acc += len(w) + 1
        if len(chunks) < n - 1 and acc >= total * (len(chunks) + 1) / n:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def fmt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_cues(start: float, duration: float, text: str, max_chars: int, max_lines: int):
    """Split the segment duration across its cues in proportion to character count."""
    parts = split_into_cues(text, max_chars, max_lines)
    total = sum(len(p.replace("\n", " ")) for p in parts) or 1
    t = start
    for p in parts:
        d = duration * len(p.replace("\n", " ")) / total
        yield t, t + d, p
        t += d


def write_srt(cues, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_time(s)} --> {fmt_time(e)}\n{text}\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate SRT subtitles with Whisper + Silero VAD (offline, CPU).")
    p.add_argument("input", nargs="?", type=Path, help="audio/video file (mp3, wav, mp4, ...)")
    p.add_argument("-o", "--output", type=Path, help="output SRT file (default: <input>.srt)")
    p.add_argument("-l", "--language", default="id", help="Whisper language code (default: id = Indonesian)")
    p.add_argument("--model", default="medium", help="Whisper size: tiny/base/small/medium (default: medium)")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--download-models", action="store_true", help="download the models and exit")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-chars", type=int, default=42, help="max characters per line (default: 42)")
    p.add_argument("--max-lines", type=int, default=2, help="max lines per subtitle (default: 2)")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--min-silence", type=float, default=0.5, help="pause (seconds) that separates segments")
    p.add_argument("--max-speech", type=float, default=20.0, help="max segment length (seconds, <30)")
    p.add_argument("-v", "--verbose", action="store_true", help="print each segment's text during transcription")
    args = p.parse_args()

    if args.download_models:
        download_models(args.model_dir, args.model)
        return
    if not args.input:
        p.error("input is required")
    if not args.input.exists():
        p.error(f"file not found: {args.input}")

    files = model_files(args.model_dir, args.model)
    if not all(f.exists() for f in files.values()):
        download_models(args.model_dir, args.model)

    output = args.output or args.input.with_suffix(".srt")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print("[1/4] Converting audio to mono 16 kHz WAV ...", file=sys.stderr)
        convert_to_wav(args.input, wav)
        samples = read_wav(wav)

    print(f"[2/4] Detecting speech (VAD) in {len(samples) / SAMPLE_RATE:.1f} s of audio ...", file=sys.stderr)
    segments = vad_segments(samples, files["vad"], args)
    print(f"      {len(segments)} speech segments", file=sys.stderr)

    print(f"[3/4] Transcribing with Whisper {args.model} (language: {args.language}) ...", file=sys.stderr)
    recognizer = create_recognizer(files, args)
    cues = []
    # Progress is measured in seconds of speech, not segment count, since segment lengths vary.
    speech_total = sum(len(seg) for _, seg in segments) / SAMPLE_RATE
    bar = ProgressBar(speech_total, "      ASR ", " s", fmt_clock)
    bar.draw()
    for i, (start, seg) in enumerate(segments, 1):
        duration = len(seg) / SAMPLE_RATE
        text = transcribe(recognizer, seg)
        bar.advance(duration)
        if args.verbose:
            bar.write(f"      [{i}/{len(segments)}] {fmt_time(start)} {text}")
        if text:
            cues.extend(build_cues(start, duration, text, args.max_chars, args.max_lines))
    bar.close()

    print(f"[4/4] Writing {len(cues)} subtitles to {output}", file=sys.stderr)
    write_srt(cues, output)
    print(f"Done in {time.time() - t0:.1f} s.", file=sys.stderr)


if __name__ == "__main__":
    main()
