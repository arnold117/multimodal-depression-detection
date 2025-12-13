# Build the dev image. Customize name/tag via parameters.
param(
    [string]$Image = "multimodal-depression:dev",
    [string]$Dockerfile = "Dockerfile",
    [switch]$NoCache
)

$context = "$PWD"

$cmd = @(
    "build",
    "-t", $Image,
    "-f", $Dockerfile
)
if ($NoCache) { $cmd += "--no-cache" }
$cmd += $context

Write-Host "Building image $Image ..." -ForegroundColor Green

docker @cmd
