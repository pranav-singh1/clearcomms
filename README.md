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

ClearComms processes radio audio through three local stages:

**1. Offline Speech Recognition**

Audio is transcribed locally using an optimized version of Whisper from OpenAI.

Engineering focus includes:

* Running Whisper with ONNX Runtime
* Model optimization and quantization
* Parameter tuning for noisy radio audio
* Low latency on device inference

Based on:
[https://github.com/thatrandomfrenchdude/simple-whisper-transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription)

---

**2. Structured Incident Extraction**

The transcript is processed by a local LLM such as LLaMA from Meta to convert raw speech into structured outputs.

Example:

**Raw transcript**

> unit 12 need backup at 5th street possible fire

**Structured output**

```
Location: 5th Street
Request: Backup
Incident: Fire
Urgency: High
```

This makes communication faster to interpret and act on.

---

**3. Offline End to End Pipeline**

```
Radio Audio
   ↓
Whisper (ONNX, on device)
   ↓
Transcript
   ↓
Local LLaMA
   ↓
Structured Incident Report
```

Everything runs fully offline.

---

## Key Features

* Fully offline operation
* Optimized Whisper inference using ONNX
* Structured incident extraction with local LLM
* Designed for Qualcomm AI hardware
* Fast, reliable transcription in noisy environments
* Simple, usable interface for real time use

---

## Tech Stack

* Python
* Whisper (ONNX Runtime)
* LLaMA 8B (local inference)
* Qualcomm AI Hub tools
* ONNX Runtime

---

## Demo

Input:

* Noisy walkie talkie audio

Output:

* Transcript
* Structured incident summary

Running locally with no internet.

---

## Hackathon Scope

This project focuses on:

* Optimizing Whisper for on device radio transcription
* Running LLaMA locally for structured outputs
* Delivering a complete offline pipeline
* Clean, usable demo experience

---

## Why It Matters

ClearComms makes critical communication understandable and actionable in the exact environments where reliability matters most.
