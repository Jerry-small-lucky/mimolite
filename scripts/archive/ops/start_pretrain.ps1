# Launch MiMo-Lite pretraining as a detached background process.
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
# when there is no BOM, so non-ASCII comments here break parsing.
#
# Usage:  powershell -ExecutionPolicy Bypass -File start_pretrain.ps1

$ErrorActionPreference = 'Stop'
$root = 'F:\' + [char]0x81EA + [char]0x7814 + [char]0x5927 + [char]0x6A21 + [char]0x578B
Set-Location $root

$trainArgs = @(
  'pretrain.py',
  '--steps', '12000',
  '--micro-bs', '8',
  '--accum', '4',
  '--seq-len', '512',
  '--lr', '1.5e-3',
  '--warmup', '200',
  '--zh-ratio', '0.7',
  '--save-every', '500',
  '--eval-every', '500',
  '--log-every', '50',
  '--run-name', 'pretrain'
)

Write-Host ("Launching: python " + ($trainArgs -join ' '))
$p = Start-Process -FilePath 'python' -ArgumentList $trainArgs `
     -WorkingDirectory $root -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput "$root\logs\pretrain_stdout.log" `
     -RedirectStandardError  "$root\logs\pretrain_stderr.log"

Write-Host ("PID = " + $p.Id)
Write-Host ("Tail log : Get-Content '$root\logs\pretrain.log' -Wait -Tail 20")
Write-Host ("Stop     : Stop-Process -Id " + $p.Id)