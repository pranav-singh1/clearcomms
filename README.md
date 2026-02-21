# ClearComms

ClearComms is a fully offline AI system that converts noisy radio communication into accurate transcripts and structured incident summaries. It is optimized to run locally on Qualcomm AI laptops with no internet connection.

The project focuses on improving transcription reliability in emergency and field communication scenarios using optimized speech recognition and local language models.

---

## Problem

First responders and field teams rely on radios that produce noisy, clipped, and hard to understand audio. This leads to:

* Misheard instructions
* Missed location or hazard details
* Slower and less effective response

Internet access is often unavailable in these environments, making cloud solutions unreliable.

ClearComms solves this by running transcription and structuring fully on device.

---

## Solution

ClearComms processes radio audio through a layered on-device pipeline. Each layer has a clear responsibility and feeds into the next.

**1. Audio Input and Preprocessing**

Raw audio is loaded and normalized before any model touches it.

* Convert to mono and resample to 16 kHz
* Normalize volume to stabilize amplitude
* Bandpass filter around 300–3400 Hz (radio bandwidth)
* Automatic gain control for consistent loudness
* Optional light noise gate to reduce constant hiss

This ensures consistent input for downstream models regardless of recording conditions.

---

**2. Speech Enhancement (Optional)**

An optional speech enhancement step can reduce additive noise such as wind, hiss, crowd, or sirens.

* Only applied when it measurably improves transcription accuracy
* Does not recover clipped audio, dropouts, or words that were never in the signal
* Can be toggled on or off depending on audio conditions

This layer is kept conditional. If Whisper already handles the noise well, it is skipped.

---

**3. Offline Speech Recognition**

Audio is transcribed locally using an optimized version of Whisper from OpenAI.

Engineering focus includes:

* Running Whisper with ONNX Runtime
* Model optimization and quantization
* Parameter tuning for noisy radio audio
* Low latency on device inference using Snapdragon NPU acceleration

Based on [simple-whisper-transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription) by thatrandomfrenchdude.

---

**4. Transcript Cleanup**

The raw transcript is cleaned up before structured extraction. This recovers intelligibility that audio enhancement alone cannot fix.

* Fix common ASR garbles (e.g. "mapple street" to "Maple Street")
* Restore punctuation and casing
* Expand abbreviations (e.g. "med" to "medical")
* Remove repeated words caused by radio jitter

Implemented with a small local LLM or a lightweight rules and dictionary fallback.

---

**5. Structured Incident Extraction**

The cleaned transcript is processed by a local LLM such as LLaMA from Meta to produce structured incident output.

Example:

**Cleaned transcript**

> Engine 12 respond to 235 Maple Street. Smoke visible. Need backup.

**Structured output**

```json
{
  "request_type": "fire",
  "urgency": "high",
  "location": "235 Maple Street",
  "hazards": ["smoke"],
  "units": ["Engine 12"],
  "actions": ["respond", "send backup"]
}
```

This makes communication faster to interpret and act on.

---

**6. Review Assist Interface**

A focused interface designed for fast verification under stress, not decoration.

* Segment list with timestamps and inline playback
* Highlighted uncertain segments that need human confirmation
* Side by side view of raw vs cleaned transcript
* Export structured incident report as JSON

---

**7. Performance and Offline Proof**

Instrumentation that proves the system works on device and meets latency requirements.

* Measures enhancement time, ASR time, and post-processing time in milliseconds
* Reports end to end latency and realtime factor
* Demonstrates full offline operation with Wi-Fi disabled
* Logs CPU and NPU utilization

---

**End to End Pipeline**

```
Radio Audio
   ↓
Preprocessing (mono, 16 kHz, bandpass, AGC)
   ↓
Enhancement (optional noise reduction)
   ↓
Whisper (ONNX, on device)
   ↓
Raw Transcript
   ↓
Transcript Cleanup (LLM or rules)
   ↓
Cleaned Transcript
   ↓
Local LLaMA → Structured Incident Report
   ↓
Review Assist UI + Performance Metrics
```

Everything runs fully offline.

---

## Key Features

* Fully offline operation with no cloud dependencies
* Audio preprocessing tuned for radio bandwidth and noise conditions
* Optimized Whisper inference using ONNX with Snapdragon NPU acceleration
* Transcript cleanup to recover intelligibility beyond what audio enhancement can fix
* Structured incident extraction with local LLM outputting actionable JSON
* Review assist interface with playback, uncertainty highlighting, and export
* Performance instrumentation with latency and utilization metrics
* Designed for Qualcomm AI hardware

---

## Tech Stack

* Python
* Whisper (ONNX Runtime)
* LLaMA 3 8B Instruct (local inference via llama-cpp-python)
* Qualcomm AI Hub tools
* ONNX Runtime
* Streamlit (UI)

---

## Demo

Input:

* Noisy walkie talkie audio clip

Output:

* Preprocessed audio (bandpass filtered, normalized)
* Raw transcript from Whisper
* Cleaned transcript with corrected spelling, punctuation, and formatting
* Structured incident JSON with type, urgency, location, hazards, units, and actions
* Performance metrics showing per-stage and end to end latency

Running locally with no internet.

---

## Hackathon Scope

This project focuses on:

* Audio preprocessing pipeline tuned for radio speech
* Optimizing Whisper for on device radio transcription
* Transcript cleanup using a local LLM or rule-based fallback
* Running LLaMA locally for structured incident extraction
* Review assist UI with playback and verification features
* Performance instrumentation proving on-device latency and offline capability
* Delivering a complete offline pipeline with clean demo experience

---

## Data Sources and References

**Audio Data**

* [LibriSpeech ASR Corpus](https://www.openslr.org/12) — Large-scale corpus of read English speech (CC BY 4.0). Used as clean speech source for generating simulated radio audio training pairs. Specifically uses the `test-clean` subset.
  * V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015
  * Original audio derived from [LibriVox](https://librivox.org/) public domain audiobooks

**Models**

* [Whisper](https://github.com/openai/whisper) — Open source speech recognition model by OpenAI. Used for on-device transcription via ONNX Runtime.
* [LLaMA](https://ai.meta.com/llama/) — Open source large language model by Meta. Used locally for transcript cleanup and structured incident extraction.

**Tools and Frameworks**

* [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Model optimization and deployment for Snapdragon hardware
* [ONNX Runtime](https://onnxruntime.ai/) — Cross-platform inference engine for optimized model execution
* [simple-whisper-transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription) — Reference implementation for Whisper transcription pipeline

**License**

* This project is licensed under GPL v3. See [LICENSE](LICENSE).
* LibriSpeech data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Why It Matters

ClearComms makes critical communication understandable and actionable in the exact environments where reliability matters most.

---

## Setup Docs

* Streamlit run/install guide: [streamline.md](streamline.md)
* Snapdragon setup checklist: [snapdragon_setup.md](snapdragon_setup.md)
* Environment diagnostics:

```bash
python tools/snapdragon_doctor.py
```
