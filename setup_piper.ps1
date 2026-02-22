# Piper TTS setup for ClearComms (Windows)
# Run from project root: .\setup_piper.ps1

$ErrorActionPreference = "Stop"
$BaseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
$VoiceDir = Join-Path (Join-Path $PSScriptRoot "assets") "voices"
$OnnxFile = "en_US-lessac-medium.onnx"
$JsonFile = "en_US-lessac-medium.onnx.json"

New-Item -ItemType Directory -Force -Path $VoiceDir | Out-Null
Set-Location $VoiceDir

Write-Host "Downloading $OnnxFile ..."
Invoke-WebRequest -Uri "$BaseUrl/$OnnxFile" -OutFile $OnnxFile -UseBasicParsing

Write-Host "Downloading $JsonFile ..."
Invoke-WebRequest -Uri "$BaseUrl/$JsonFile" -OutFile $JsonFile -UseBasicParsing

$ModelPath = (Get-Item $OnnxFile).FullName
Write-Host ""
Write-Host "Done. Voice files saved to: $VoiceDir"
Write-Host ""
Write-Host "Before starting the backend, set (in the same PowerShell):"
Write-Host "  `$env:PIPER_MODEL_PATH=`"$ModelPath`""
Write-Host ""
Write-Host "If piper is not on PATH, also set:"
Write-Host '  $env:PIPER_BIN="C:\path\to\piper.exe"'
Write-Host ""
Write-Host "Then start the backend:"
Write-Host "  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001"
