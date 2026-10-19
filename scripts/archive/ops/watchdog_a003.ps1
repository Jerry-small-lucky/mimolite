# Watchdog for the a003 ablation suite: guarantees all 6 configs reach 1200 steps.
# ASCII-only on purpose: PowerShell 5.1 fails to parse non-ASCII .ps1 files.
# Root is derived from this script's own location so no non-ASCII literal appears here.
#
# Design note -- why this is a blocking loop and not "detect, then relaunch":
# the first version polled WMI for a running ablate.py and relaunched when the
# query came back empty. WMI failed once under load, the watchdog read that as
# "nothing is running", and it started a SECOND trainer for the same run dir:
# two processes then wrote runs\a003_p_mid\loss_curve.json concurrently, which
# interleaved the step indices. Here the child is started with -Wait, so exactly
# one trainer can ever exist and detection is never needed.
#
# Progress is NOT resumed mid-run: ablate.py --skip-done reruns any incomplete
# config from scratch, which is what keeps the token budget honest.

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py       = 'C:\Python314\python.exe'
$LogDir   = Join-Path $Root 'logs'
$WdLog    = Join-Path $LogDir 'watchdog_a003.log'
$LockFile = Join-Path $LogDir 'watchdog_a003.lock'
$Suite    = @('a003_r13', 'a003_r6', 'a003_r25', 'a003_p_bot', 'a003_p_mid', 'a003_p_top')
$Steps    = 1200
$MaxHours = 6
$MaxTries = 4

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Say([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $WdLog -Value $line -Encoding ASCII
}

# Refuse to start a second supervisor: two would race to train the same run dir.
if (Test-Path $LockFile) {
    $owner = [int](Get-Content $LockFile -ErrorAction SilentlyContinue)
    if ($owner -gt 0 -and (Get-Process -Id $owner -ErrorAction SilentlyContinue)) {
        Say "another watchdog already owns this suite (PID $owner); exiting"
        exit 0
    }
}
Set-Content -Path $LockFile -Value $PID -Encoding ASCII

function Get-Incomplete {
    <#
      A config counts as done only when its last val record reaches the target step.
      Same criterion ablate.py uses for --skip-done, so the watchdog and the
      suite runner never disagree about what still needs work.
    #>
    $bad = @()
    foreach ($n in $Suite) {
        $lc = Join-Path $Root ("runs\{0}\loss_curve.json" -f $n)
        if (-not (Test-Path $lc)) { $bad += $n; continue }
        try {
            $d = Get-Content $lc -Raw -Encoding UTF8 | ConvertFrom-Json
            $v = $d.val
            if (-not $v -or $v.Count -eq 0) { $bad += $n; continue }
            if ($v[$v.Count - 1].step -lt $Steps) { $bad += $n }
        } catch {
            $bad += $n     # unreadable/half-written JSON: the rerun rewrites it
        }
    }
    return $bad
}

# Force UTF-8 on every child. Without this the suite dies on its very first
# print of the tick character under a GBK console (UnicodeEncodeError).
$env:PYTHONIOENCODING  = 'utf-8'
$env:PYTHONUTF8       = '1'
$env:PYTHONUNBUFFERED = '1'

$deadline = (Get-Date).AddHours($MaxHours)
$tries    = @{}
foreach ($n in $Suite) { $tries[$n] = 0 }
$attempt  = 0

Say ("watchdog start PID={0} (suite=a003, {1} configs x {2} steps, deadline {3})" -f `
        $PID, $Suite.Count, $Steps, $deadline.ToString('HH:mm:ss'))

while ($true) {
    $bad = Get-Incomplete
    if ($bad.Count -eq 0) {
        Say ("ALL DONE: all {0} configs reached {1} steps" -f $Suite.Count, $Steps)
        exit 0
    }

    $exhausted = @($bad | Where-Object { $tries[$_] -ge $MaxTries })
    if ($exhausted.Count -gt 0) {
        Say ("ABORT: {0} hit {1} failed attempts; manual intervention needed" -f ($exhausted -join ','), $MaxTries)
        exit 1
    }

    if ((Get-Date) -gt $deadline) {
        Say ("DEADLINE EXCEEDED after {0} h; still missing: {1}" -f $MaxHours, ($bad -join ','))
        exit 1
    }

    $attempt++
    foreach ($n in $bad) { $tries[$n]++ }
    $runLog = Join-Path $LogDir ("ablate_try{0}.log" -f $attempt)
    $runErr = Join-Path $LogDir ("ablate_try{0}.err" -f $attempt)

    Say ("attempt {0}/{1}: running suite for {2} -> stdout={3}" -f $attempt, $MaxTries, ($bad -join ','), $runLog)

    # -Wait: block until the whole suite finishes, so a second trainer is impossible.
    $p = Start-Process -FilePath $Py `
        -ArgumentList @('ablate.py', '--suite', 'a003', '--steps', "$Steps", '--skip-done') `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $runLog `
        -RedirectStandardError  $runErr `
        -WindowStyle Hidden -PassThru -Wait

    Say ("attempt {0} exited rc={1}; re-checking..." -f $attempt, $p.ExitCode)
}