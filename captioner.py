#!/usr/bin/env python3
"""Buat subtitle SRT dari file audio/video secara lokal.

Pipeline:
  1. ffmpeg      -> WAV mono 16 kHz
  2. Silero VAD  -> potongan ucapan (sumber timestamp)
  3. Whisper     -> transkripsi tiap potongan (sherpa-onnx, int8, CPU)
  4. SRT         -> maks 2 baris per subtitle, durasi dibagi sesuai jumlah karakter
"""

import argparse
import os
import re
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
        print(f"Mengunduh {path.name} ...", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        for attempt in range(10):
            try:
                fetch_resumable(url, tmp)
                break
            except OSError as e:
                print(f"  koneksi terputus ({e}), mencoba lagi ...", file=sys.stderr)
                time.sleep(2)
        else:
            sys.exit(f"Gagal mengunduh {url}")
        tmp.rename(path)


def fetch_resumable(url: str, dst: Path) -> None:
    """Unduh `url` ke `dst`, melanjutkan dari ukuran file yang sudah ada."""
    offset = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if offset and resp.status == 206 else "wb"
        with open(dst, mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)


# ---------------------------------------------------------------------------
# 1. Konversi audio
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
# 2. Pemotongan dengan VAD
# ---------------------------------------------------------------------------

def vad_segments(samples: np.ndarray, vad_model: Path, args) -> list[tuple[float, np.ndarray]]:
    """Kembalikan list (waktu_mulai_detik, sampel) untuk tiap potongan ucapan."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence
    config.silero_vad.min_speech_duration = 0.25
    # Whisper hanya menerima maks 30 detik per input.
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

    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        drain()
    vad.flush()
    drain()
    return segments


# ---------------------------------------------------------------------------
# 3. Transkripsi
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
# 4. Pembuatan SRT
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
    """Pecah teks menjadi cue berisi maks `max_lines` baris x `max_chars` karakter.

    Batas kalimat diutamakan; kalimat yang terlalu panjang dibungkus per kata,
    lalu dikelompokkan per `max_lines` baris.
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
    """Bagi kata menjadi `n` kelompok dengan jumlah karakter yang kira-kira sama."""
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
    """Bagi durasi potongan ke tiap cue secara proporsional dengan jumlah karakter."""
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
    p = argparse.ArgumentParser(description="Buat subtitle SRT dengan Whisper + Silero VAD (offline, CPU).")
    p.add_argument("input", nargs="?", type=Path, help="file audio/video (mp3, wav, mp4, ...)")
    p.add_argument("-o", "--output", type=Path, help="file SRT keluaran (default: <input>.srt)")
    p.add_argument("-l", "--language", default="id", help="kode bahasa Whisper (default: id)")
    p.add_argument("--model", default="medium", help="ukuran Whisper: tiny/base/small/medium (default: medium)")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--download-models", action="store_true", help="unduh model lalu keluar")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-chars", type=int, default=42, help="maks karakter per baris (default: 42)")
    p.add_argument("--max-lines", type=int, default=2, help="maks baris per subtitle (default: 2)")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--min-silence", type=float, default=0.5, help="jeda (detik) pemisah potongan")
    p.add_argument("--max-speech", type=float, default=20.0, help="panjang maks potongan (detik, <30)")
    args = p.parse_args()

    if args.download_models:
        download_models(args.model_dir, args.model)
        return
    if not args.input:
        p.error("input wajib diisi")
    if not args.input.exists():
        p.error(f"file tidak ditemukan: {args.input}")

    files = model_files(args.model_dir, args.model)
    if not all(f.exists() for f in files.values()):
        download_models(args.model_dir, args.model)

    output = args.output or args.input.with_suffix(".srt")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print("[1/4] Konversi audio ke WAV mono 16 kHz ...", file=sys.stderr)
        convert_to_wav(args.input, wav)
        samples = read_wav(wav)

    print(f"[2/4] Deteksi ucapan (VAD) pada {len(samples) / SAMPLE_RATE:.1f} detik audio ...", file=sys.stderr)
    segments = vad_segments(samples, files["vad"], args)
    print(f"      {len(segments)} potongan ucapan", file=sys.stderr)

    print(f"[3/4] Transkripsi dengan Whisper {args.model} (bahasa: {args.language}) ...", file=sys.stderr)
    recognizer = create_recognizer(files, args)
    cues = []
    for i, (start, seg) in enumerate(segments, 1):
        duration = len(seg) / SAMPLE_RATE
        text = transcribe(recognizer, seg)
        print(f"      [{i}/{len(segments)}] {fmt_time(start)} {text}", file=sys.stderr)
        if text:
            cues.extend(build_cues(start, duration, text, args.max_chars, args.max_lines))

    print(f"[4/4] Menulis {len(cues)} subtitle ke {output}", file=sys.stderr)
    write_srt(cues, output)
    print(f"Selesai dalam {time.time() - t0:.1f} detik.", file=sys.stderr)


if __name__ == "__main__":
    main()
