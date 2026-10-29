# docker_scan.ps1 — One face scan with the Docker image on this Windows PC, photos and results on the local disk.
# Nothing goes through Cloudflare R2: without JOB_ID the container runs the pipeline on the mounted folders.
#
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name alin
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name alin -Extra "--no-spectacles","--texture-size","2048"
#   powershell -ExecutionPolicy Bypass -File tools\docker_scan.ps1 -Name alin -InputDir D:\scanari\alin
#
# Results: <repo>\teste\<Name>\ (OBJ, GLB, STL, 3MF, object.json, stats.json, photogrammetry_manifest.json).
# Docker Desktop: the default 4096 px atlas needs ~10 GB of RAM in the Docker VM; otherwise --texture-size 2048.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$InputDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "input_images"),
    [string]$OutputDir = "",
    [string]$Image = "face-scan:local",
    [string[]]$Extra = @()
)
if (-not $OutputDir) { $OutputDir = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) "teste") $Name }
if (-not (Test-Path (Join-Path $InputDir "rig"))) { Write-Error "No rig\0 and rig\1 in $InputDir"; exit 1 }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$args_main = @("--rig-dir", "/app/input_images/rig", "-o", "/app/output", "--name", "model") + $Extra
Write-Host "Scan $Name : $InputDir -> $OutputDir ($Image) $($Extra -join ' ')"
$t0 = Get-Date
docker run --rm `
    -v "${InputDir}:/app/input_images:ro" `
    -v "${OutputDir}:/app/output" `
    $Image @args_main
$code = $LASTEXITCODE
Write-Host ("Done in {0:N1} min, exit code {1}. Results: {2}" -f ((Get-Date) - $t0).TotalMinutes, $code, $OutputDir)
exit $code
