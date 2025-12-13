# Convenience script to start the dev container with code + data mounted
# Edit $DataPath if your data lives elsewhere.

param(
    [string]$DataPath = "$PWD\data",
    [string]$ContainerName = "mdd-dev",
    [int]$Port = 8888
)

$Image = "multimodal-depression:dev"

# Ensure the image exists
Write-Host "Checking image $Image ..."
$exists = docker images --format '{{.Repository}}:{{.Tag}}' | Where-Object { $_ -eq $Image }
if (-not $exists) {
    Write-Host "Image not found. Building..." -ForegroundColor Yellow
    docker build -t $Image .
}

Write-Host "Starting container $ContainerName ..." -ForegroundColor Green

# Run container with mounts; remove on exit for a clean slate
$cmd = @(
    "run", "-it", "--name", $ContainerName, "--rm",
    "-v", "$PWD:/app",
    "-v", "$DataPath:/app/data",
    "-p", "$Port:8888",
    $Image
)

docker @cmd
