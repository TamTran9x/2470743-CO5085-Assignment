# Everything that needs the GPU, in order; safe to re-launch (each step skips or resumes finished work).
#   powershell -ExecutionPolicy Bypass -File scripts\post_train.ps1
# On battery the laptop GPU is capped at ~10 W (20x slower, and timings become meaningless),
# so every step first waits until the machine is on AC power.
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
Set-Location (Join-Path $PSScriptRoot '..')

function Wait-AC {
    $told = $false
    while ($true) {
        $s = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue
        if (-not $s -or $s.PowerOnline) { return }   # desktop PC (no battery) or plugged in
        if (-not $told) { "waiting for AC power $(Get-Date -Format 'HH:mm:ss')"; $told = $true }
        Start-Sleep -Seconds 20
    }
}

"pipeline started $(Get-Date -Format 'HH:mm:ss')"
Wait-AC
if (-not (Test-Path results\benchmark.json)) { & $py -W ignore -m scripts.benchmark --crops 512 --batch-size 8 --json results\benchmark.json }
Wait-AC
& $py -W ignore -m src.erf --runs deeplabv3plus_r50 resnet50_mlp segformer_b0 segformer_b2
Wait-AC
& $py -W ignore -m scripts.run_all
"training done $(Get-Date -Format 'HH:mm:ss')"
Wait-AC
& $py -W ignore -m src.robustness
& $py -W ignore -m src.analyze
"pipeline finished $(Get-Date -Format 'HH:mm:ss')"
