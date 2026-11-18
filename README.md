# ClipEngine

Local GPU-accelerated vertical short-form video clipper, transcription pipeline, and permission-gated RSS source scout.

## Features
- Local GPU Rendering: Vertical video framing (Cinema Blur and Gaming Split) powered by FFmpeg and NVIDIA CUDA.
- Automated Transcription & Scoring: Local Whisper transcription paired with Gemini hook detection and virality scoring.
- Permission-Gated Source Scout: Monitors YouTube RSS feeds for confirmed creator channels, filters out Shorts and livestreams, and prevents duplicate processing.
- Copy-Ready Upload Packs: Generates YouTube Shorts titles, descriptions, hashtags, and discussion questions for one-click studio scheduling.

## Prerequisites
- Windows 10/11 or Linux with Python 3.10+
- FFmpeg installed and available on your system PATH
- NVIDIA GPU with CUDA support (recommended)
- A free Google Gemini API key from Google AI Studio

## Quick Start

    git clone https://github.com/ar-ghouri/clipengine.git
    cd clipengine
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env

Open .env, paste your GEMINI_API_KEY, then start the local studio:

    python server.py

Open http://127.0.0.1:8000 in your browser.

## Responsible Use
Only clip and publish content you own or have explicit creator permission or license rights to reuse.
