# Auto pipeline: wait for pretrain, then benchmark and run ablation A-003.
# Pure ASCII on purpose: PowerShell 5.1 fails to parse non-ASCII .ps1 files.
# The project root is derived from this script's own location so that no
# non-ASCII literal ever appears in the source.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'pipeline.log'

function Say([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding ASCII
}

Say "pipeline start"

# ---- 1. wait for the running pretrain (PID 16244) ----
if (Get-Process -Id 16244 -ErrorAction SilentlyContinue) {
    Say "pretrain PID 16244 still running, waiting ..."
    while (Get-Process -Id 16244 -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Say "pretrain PID 16244 exited"
} else {
    Say "pretrain PID 16244 not found (already finished or not started)"
}

# give the GPU a moment to release memory
Start-Sleep -Seconds 30

$final = Join-Path $Root 'runs\pretrain\final.pt'
if (Test-Path $final) {
    $sz = [math]::Round((Get-Item $final).Length / 1MB, 1)
    Say "final.pt present ($sz MB)"
} else {
    Say "WARNING: final.pt missing, aborting pipeline"
    exit 1
}

# ---- 2. throughput benchmark: grad checkpoint on vs off ----
Say "running bench_train.py"
& python bench_train.py *>> $Log
Say "bench done (see $Log)"

Start-Sleep -Seconds 20

# ---- 3. ablation A-003: SWA/GA ratio and placement ----
Say "running ablation suite a003 (6 configs x 1200 steps)"
& python ablate.py --suite a003 --steps 1200 *>> $Log
Say "a003 done"

Say "pipeline finished"
