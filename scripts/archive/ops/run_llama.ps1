# 用完全独立的方式跑 llama-cli（不挂在当前进程组内，避免被外部中断）
# ASCII-only: PowerShell 5.1 reads .ps1 as ANSI without BOM.
param(
  [string]$Model,
  [string]$Prompt = 'test',
  [string]$N = '20',
  [string]$Temp = '0',
  [string]$Out = 'F:\mimolite_llama_out.txt'
)

$exe = 'F:\AI\tools\llama.cpp\llama-cli.exe'
$argList = @('-m', $Model, '-p', $Prompt, '-n', $N, '--temp', $Temp,
             '--seed', '1', '-ngl', '99', '--no-display-prompt')

$p = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru `
     -RedirectStandardOutput $Out -RedirectStandardError "$Out.err"
Write-Host ("started pid=" + $p.Id)
$p.WaitForExit()
Write-Host ("exit=" + $p.ExitCode + "  bytes=" + (Get-Item $Out).Length)