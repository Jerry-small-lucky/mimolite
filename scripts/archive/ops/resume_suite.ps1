# Resume-driven supervisor for the remaining a003 configs.
# ASCII-only on purpose: PowerShell 5.1 fails to parse non-ASCII .ps1 files.
#
# Why not ablate.py --skip-done: that reruns an incomplete config FROM SCRATCH.
# This machine is thermally saturated (nvlddmkm event 153 TDR kills a trainer
# every 6-13 min, SW Thermal Slowdown active), and a 1200-step config needs
# ~22 min. Restarting from zero therefore never converges: every attempt was
# losing all of its progress. pretrain.py now stores the sampler rng + torch +
# numpy rng state in the checkpoint, so a kill costs at most --save-every steps.
#
# Config args are copied verbatim from the command line ablate.py builds, so the
# rerun stays protocol-identical to the 4 configs that already completed.
# Only --save-every changes (600 -> 200); checkpoint cadence has no effect on
# the training trajectory, it just shrinks how much work a TDR can destroy.

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py       = 'C:\Python314\python.exe'
$LogDir   = Join-Path $Root 'logs'
$Log      = Join-Path $LogDir 'resume_suite.log'
$LockFile = Join-Path $LogDir 'resume_suite.lock'
$Suite    = @('a003_p_mid', 'a003_p_top')
$GaLayers = @{ 'a003_p_mid' = '6,7'; 'a003_p_top' = '7,13' }
$Steps    = 1200
$SaveEvery = 200
$CooldownS = 75
$MaxHours = 6

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Say([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding ASCII
}

if (Test-Path $LockFile) {
    $owner = [int](Get-Content $LockFile -ErrorAction SilentlyContinue)
    if ($owner -gt 0 -and (Get-Process -Id $owner -ErrorAction SilentlyContinue)) {
        Say "another supervisor already owns the suite (PID $owner); exiting"
        exit 0
    }
}
Set-Content -Path $LockFile -Value $PID -Encoding ASCII

function Get-LastValStep([string]$name) {
    $lc = Join-Path $Root ("runs\{0}\loss_curve.json" -f $name)
    if (-not (Test-Path $lc)) { return -1 }
    try {
        $d = Get-Content $lc -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $d.val -or $d.val.Count -eq 0) { return -1 }
        return [int]$d.val[$d.val.Count - 1].step
    } catch { return -1 }
}

function Test-Complete([string]$name) {
    if ((Get-LastValStep $name) -lt $Steps) { return $false }
    return (Test-Path (Join-Path $Root ("runs\{0}\final.pt" -f $name)))
}

$env:PYTHONIOENCODING  = 'utf-8'
$env:PYTHONUTF8       = '1'
$env:PYTHONUNBUFFERED = '1'

$deadline = (Get-Date).AddHours($MaxHours)
Say ("supervisor start PID={0} (targets: {1}, deadline {2})" -f `
        $PID, ($Suite -join ','), $deadline.ToString('HH:mm:ss'))

foreach ($cfg in $Suite) {
    $attempt = 0
    while (-not (Test-Complete $cfg)) {
        if ((Get-Date) -gt $deadline) {
            Say ("DEADLINE EXCEEDED; {0} stuck at step {1}" -f $cfg, (Get-LastValStep $cfg))
            exit 1
        }

        $attempt++
        $ckpt = Join-Path $Root ("runs\{0}\last.pt" -f $cfg)
        $resume = Test-Path $ckpt
        $before = Get-LastValStep $cfg

        $args = @(
            'pretrain.py',
            '--run-name', $cfg,
            '--steps', "$Steps",
            '--seq-len', '512',
            '--micro-bs', '8',
            '--accum', '4',
            '--lr', '0.0015',
            '--n-layer', '14',
            '--d-model', '768',
            '--n-head', '12',
            '--d-ff', '2048',
            '--sched', 'cosine',
            '--gradpower-p', '1.0',
            '--ga-layers', $GaLayers[$cfg],
            '--eval-every', '300',
            '--save-every', "$SaveEvery",
            '--seed', '1234',
            '--wsd-decay-ratio', '0.2',
            '--attn-impl', 'dense'
        )
        if ($resume) { $args += '--resume' }

        $tag = if ($resume) { "resume@{0}" -f $before } else { 'fresh' }
        $runLog = Join-Path $LogDir ("resume_{0}_a{1}.log" -f $cfg, $attempt)
        $runErr = Join-Path $LogDir ("resume_{0}_a{1}.err" -f $cfg, $attempt)

        Say ("{0} attempt {1} [{2}] last_val_step={3} -> {4}" -f $cfg, $attempt, $tag, $before, $runLog)

        $p = Start-Process -FilePath $Py -ArgumentList $args `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $runLog `
            -RedirectStandardError  $runErr `
            -WindowStyle Hidden -PassThru -Wait

        $after = Get-LastValStep $cfg
        Say ("{0} attempt {1} rc={2}; last_val_step {3} -> {4}{5}" -f `
             $cfg, $attempt, $p.ExitCode, $before, $after,
             $(if (Test-Complete $cfg) { ' [COMPLETE]' } else { '' }))

        # A checkpoint written before the rng fix cannot be resumed honestly;
        # pretrain.py refuses it with a RESUME_REFUSED marker on stderr.
        # Move it aside so the next attempt starts clean. Both streams are
        # checked because SystemExit writes to stderr, not stdout.
        $refused = (Test-Path $runErr) -and (Select-String -Path $runErr -Pattern 'RESUME_REFUSED' -Quiet)
        if (-not $refused -and (Test-Path $runLog)) {
            $refused = Select-String -Path $runLog -Pattern 'RESUME_REFUSED' -Quiet
        }
        if ($refused) {
            $stale = Join-Path $Root ("runs\{0}\last.stale.pt" -f $cfg)
            if (Test-Path $ckpt) {
                Move-Item $ckpt $stale -Force
                Say ("{0}: moved pre-fix checkpoint aside (last.stale.pt)" -f $cfg)
            }
        }

        if (-not (Test-Complete $cfg)) {
            Say ("{0}: killed before finishing; cooling down {1}s" -f $cfg, $CooldownS)
            Start-Sleep -Seconds $CooldownS
        }
    }
    Say ("{0} COMPLETE at {1} steps" -f $cfg, $Steps)
}

Say "ALL TARGETS COMPLETE"
exit 0