# closed_captioner

Membuat subtitle `.srt` dari audio/video secara lokal di CPU. Audio tidak dikirim ke layanan transkripsi pihak ketiga.

Transkripsinya dibuat dengan **Whisper medium** (OpenAI), dijalankan lewat **sherpa-onnx**, yaitu versi ONNX yang dikuantisasi ke int8.

## Pipeline

1. **Konversi audio**: `ffmpeg` mengubah input menjadi WAV mono 16 kHz.
2. **Pemotongan dengan VAD**: Silero VAD memotong audio per bagian ucapan. Timestamp subtitle diambil dari potongan ini.
3. **Transkripsi**: tiap potongan ditranskripsi dengan bahasa `id` (Indonesia).
4. **Pembuatan SRT**: kalimat panjang dipecah menjadi subtitle maksimal 2 baris, dan durasi potongan dibagi sesuai jumlah karakter.

## Instalasi

Butuh Python 3.10+ dan `ffmpeg` (`brew install ffmpeg`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python captioner.py --download-models   # ~1 GB ke ./models
```

## Pemakaian

```bash
.venv/bin/python captioner.py rekaman.mp3              # -> rekaman.srt
.venv/bin/python captioner.py video.mp4 -o sub.srt
.venv/bin/python captioner.py audio.mp3 -l en --model small --max-chars 37
```

| Opsi | Default | Keterangan |
|---|---|---|
| `-l, --language` | `id` | Kode bahasa Whisper |
| `--model` | `medium` | `tiny` / `base` / `small` / `medium` |
| `--max-chars` | `42` | Karakter maksimal per baris |
| `--max-lines` | `2` | Baris maksimal per subtitle |
| `--min-silence` | `0.5` | Jeda (detik) yang memisahkan potongan VAD |
| `--max-speech` | `20` | Panjang maksimal potongan (Whisper dibatasi 30 detik) |
| `--threads` | CPU−1 | Thread untuk inferensi |
