<#
Bundle what the lab server cannot get from git: the built frontend (the server
has no Node.js) and the model checkpoints (Meta's download site is blocked
there, and they are not all in git). Run on this PC from the repo root:

    .\deploy\pack.ps1              # builds the frontend, writes iseg-bundle.tar.gz
    .\deploy\pack.ps1 -NoBuild     # reuse frontend\dist as it is

Then copy it over and unpack it into the app folder (deploy/ailab26.md, step 3).
model_weights\SHA256SUMS goes in too, for `sha256sum -c` on the server.
#>
param([string]$Out = 'iseg-bundle.tar.gz', [switch]$NoBuild)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $NoBuild) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue) -and (Test-Path 'C:\Program Files\nodejs')) {
        $env:PATH = "C:\Program Files\nodejs;$env:PATH"
    }
    Push-Location frontend
    try { npm run build; if ($LASTEXITCODE) { throw "frontend build failed" } } finally { Pop-Location }
}
if (-not (Test-Path frontend\dist\index.html)) { throw "frontend\dist is missing; run without -NoBuild" }

$weights = Get-ChildItem model_weights -File | Where-Object { $_.Extension -in '.pth', '.pt' } | Sort-Object Name
if (-not $weights) { throw "no checkpoints in model_weights\" }
# sha256sum format, LF line endings, so `sha256sum -c` reads it on Linux.
$sums = ($weights | ForEach-Object { "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $($_.Name)" }) -join "`n"
[IO.File]::WriteAllText((Join-Path (Get-Location) 'model_weights\SHA256SUMS'), "$sums`n")

$files = @('frontend/dist', 'model_weights/SHA256SUMS') + ($weights | ForEach-Object { "model_weights/$($_.Name)" })
tar -czf $Out @files
if ($LASTEXITCODE) { throw "tar failed" }
$size = '{0:N0} MB' -f ((Get-Item $Out).Length / 1MB)
Write-Host "$Out ($size):"
$weights | ForEach-Object { Write-Host "  model_weights/$($_.Name)" }
Write-Host "  frontend/dist"
Write-Host "SHA-256 $((Get-FileHash $Out -Algorithm SHA256).Hash.ToLower())"
