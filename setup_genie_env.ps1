# scripts/setup_genie_env.ps1
param(
  [string]$QAIRT_HOME = "C:\Users\hackathon user\Downloads\v2.43.1.260218\qairt\2.43.1.260218",
  [string]$HEXAGON_VER = "v73"   # change to v75 if needed
)

$env:GENIE_BUNDLE_DIR = 'C:\Users\hackathon user\documents\qualhack\simple-whisper-transcription\genie_bundle\llama_v3_2_3b_instruct-genie-w4a16-qualcomm_snapdragon_x_elite'
$env:DEEPGRAM_API_KEY="your-api-key-here"
$env:ENABLE_LLAMA_REVISION = "1" 
$env:QAIRT_HOME = $QAIRT_HOME
$env:Path = "$env:QAIRT_HOME\bin\aarch64-windows-msvc;$env:Path"
$env:Path = "$env:QAIRT_HOME\lib\aarch64-windows-msvc;$env:Path"
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-$HEXAGON_VER\unsigned"

Write-Host "QAIRT_HOME=$env:QAIRT_HOME"
Write-Host "ADSP_LIBRARY_PATH=$env:ADSP_LIBRARY_PATH"
Write-Host "Genie on PATH? " -NoNewline
try { (Get-Command genie-t2t-run.exe).Source | Write-Host } catch { Write-Host "NO" }