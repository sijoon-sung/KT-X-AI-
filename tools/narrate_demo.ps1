param([Parameter(Mandatory=$true)][string]$ScriptFile,[Parameter(Mandatory=$true)][string]$Out)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$spec=Get-Content -LiteralPath $ScriptFile -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice('Microsoft Heami Desktop')
$voice.Rate=1
$voice.Volume=100
try {
  for($i=0;$i -lt $spec.scenes.Count;$i++) {
    $dest=Join-Path $Out ('voice-'+$i+'.wav')
    $voice.SetOutputToWaveFile($dest)
    $voice.Speak(($spec.scenes[$i].lines -join ' '))
    $voice.SetOutputToNull()
  }
} finally {$voice.Dispose()}
