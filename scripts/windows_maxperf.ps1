<#
.SYNOPSIS
  Get the most out of an NVIDIA card on Windows before/while mining.

.DESCRIPTION
  Raises the power limit and locks clocks where the driver allows it (GeForce cards
  reject some of these - the script reports that instead of failing), raises the
  miner's process priority, and can extend the driver timeout that would otherwise
  kill long compute kernels on a display GPU.

  Every change that needs administrator rights says so, and the registry change is
  opt-in via -SetTdrDelay.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows_maxperf.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows_maxperf.ps1 -SetTdrDelay
  powershell -ExecutionPolicy Bypass -File scripts\windows_maxperf.ps1 -Monitor
#>

param(
    [switch]$SetTdrDelay,   # extend the driver TDR timeout (admin, needs a reboot)
    [switch]$Monitor,       # keep printing temperature, clocks and power
    [int]$TdrDelaySeconds = 20
)

$ErrorActionPreference = 'Continue'

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Smi {
    param([string[]]$SmiArgs)
    $out = & nvidia-smi @SmiArgs 2>&1
    [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Output = ($out -join ' ').Trim() }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi not found - install the NVIDIA driver." -ForegroundColor Red
    exit 1
}

Write-Host "== GPUs ==" -ForegroundColor Cyan
& nvidia-smi --query-gpu=index,name,driver_version,power.limit,power.max_limit,clocks.max.graphics `
             --format=csv

$indices = (& nvidia-smi --query-gpu=index --format=csv,noheader) | ForEach-Object { $_.Trim() }
$admin = Test-Admin
if (-not $admin) {
    Write-Host "`nNot running as administrator: power limit and clock locks will likely be refused." `
        -ForegroundColor Yellow
}

Write-Host "`n== power limit ==" -ForegroundColor Cyan
foreach ($i in $indices) {
    $max = (& nvidia-smi -i $i --query-gpu=power.max_limit --format=csv,noheader).Split(' ')[0]
    $res = Invoke-Smi @('-i', $i, '-pl', $max)
    if ($res.Ok) { Write-Host "  GPU $i -> $max W" -ForegroundColor Green }
    else { Write-Host "  GPU $i : not settable ($($res.Output))" -ForegroundColor Yellow }
}

Write-Host "`n== clock lock ==" -ForegroundColor Cyan
foreach ($i in $indices) {
    $maxClock = (& nvidia-smi -i $i --query-gpu=clocks.max.graphics --format=csv,noheader).Split(' ')[0]
    $res = Invoke-Smi @('-i', $i, '-lgc', "0,$maxClock")
    if ($res.Ok) { Write-Host "  GPU $i -> up to $maxClock MHz" -ForegroundColor Green }
    else { Write-Host "  GPU $i : not settable ($($res.Output))" -ForegroundColor Yellow }
}

Write-Host "`n== miner process priority ==" -ForegroundColor Cyan
$procs = Get-Process -Name 'hcminer-gpu' -ErrorAction SilentlyContinue
if ($procs) {
    foreach ($p in $procs) {
        try {
            $p.PriorityClass = [Diagnostics.ProcessPriorityClass]::High
            Write-Host "  pid $($p.Id) -> High" -ForegroundColor Green
        } catch {
            Write-Host "  pid $($p.Id) : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  hcminer-gpu is not running (start it, then re-run this script)"
}

if ($SetTdrDelay) {
    Write-Host "`n== driver timeout (TDR) ==" -ForegroundColor Cyan
    if (-not $admin) {
        Write-Host "  needs an elevated PowerShell - right-click, Run as administrator" -ForegroundColor Red
    } else {
        $key = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
        $old = (Get-ItemProperty -Path $key -Name TdrDelay -ErrorAction SilentlyContinue).TdrDelay
        Write-Host "  current TdrDelay: $(if ($null -eq $old) { 'not set (default 2 s)' } else { "$old s" })"
        Set-ItemProperty -Path $key -Name TdrDelay -Value $TdrDelaySeconds -Type DWord
        Set-ItemProperty -Path $key -Name TdrDdiDelay -Value $TdrDelaySeconds -Type DWord
        Write-Host "  set to $TdrDelaySeconds s - REBOOT for it to take effect" -ForegroundColor Green
        Write-Host "  (undo: Remove-ItemProperty -Path '$key' -Name TdrDelay,TdrDdiDelay)"
    }
}

Write-Host "`n== not automatable from here ==" -ForegroundColor Cyan
Write-Host "  NVIDIA Control Panel -> Manage 3D settings -> Power management mode ->"
Write-Host "    'Prefer maximum performance'"
Write-Host "  Keep the miner off the GPU driving your monitors if you have a second card."
Write-Host "  Core/memory overclocking is a separate tool (e.g. MSI Afterburner) and is"
Write-Host "  on you: an unstable overclock produces wrong hashes, not just crashes."

if ($Monitor) {
    Write-Host "`n== monitoring (Ctrl+C to stop) ==" -ForegroundColor Cyan
    while ($true) {
        $row = & nvidia-smi --query-gpu=index,utilization.gpu,clocks.current.graphics,temperature.gpu,power.draw `
                            --format=csv,noheader
        Write-Host ("{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), ($row -join ' | '))
        Start-Sleep -Seconds 5
    }
}
