# Run any llama.cpp exe detached and write output to a file.
# ASCII-only (PowerShell 5.1 reads .ps1 as ANSI without BOM).
param(
  [string]$Exe = 'F:\AI\tools\llama.cpp\llama-cli.exe',
  [string]$ArgLine = '',
  [string]$Out = 'F:\mimolite_tool_out.txt',
  [string]$WorkDir = ''
)

if ($WorkDir -eq '') { $WorkDir = Split-Path -Parent $Out }
$p = Start-Process -FilePath $Exe -ArgumentList $ArgLine -NoNewWindow -PassThru `
     -WorkingDirectory $WorkDir `
     -RedirectStandardOutput $Out -RedirectStandardError "$Out.err"
$p.WaitForExit()
Write-Host ("exit=" + $p.ExitCode)
Write-Host ("out_bytes=" + (Get-Item $Out).Length)
Write-Host ("err_bytes=" + (Get-Item "$Out.err").Length)