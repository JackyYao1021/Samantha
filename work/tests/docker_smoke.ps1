param([switch]$SkipBuild)

$ErrorActionPreference = 'Stop'
$taskRepoDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskDockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
if ($taskDockerCommand) {
    $taskDockerCli = $taskDockerCommand.Source
} else {
    $taskDockerCli = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
}
if (-not (Test-Path -LiteralPath $taskDockerCli)) {
    throw 'Docker CLI is not installed or is not on PATH.'
}

function Invoke-TaskDocker {
    param([string[]]$DockerArguments)
    & $taskDockerCli @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed (exit $LASTEXITCODE): $($DockerArguments -join ' ')"
    }
}

Push-Location -LiteralPath $taskRepoDir
try {
    Invoke-TaskDocker -DockerArguments @('info', '--format', '{{.ServerVersion}}')
    Invoke-TaskDocker -DockerArguments @('compose', 'config', '--quiet')
    if ($SkipBuild) {
        Invoke-TaskDocker -DockerArguments @('compose', 'up', '-d')
    } else {
        Invoke-TaskDocker -DockerArguments @('compose', 'up', '-d', '--build')
    }
    Invoke-TaskDocker -DockerArguments @('compose', 'exec', '-T', 'oe', 'python3', '-m', 'unittest', 'discover', '-s', '/work/tests', '-v')
    Invoke-TaskDocker -DockerArguments @('compose', 'exec', '-T', 'oe', 'bash', '/work/tests/shell_smoke.sh')
    Invoke-TaskDocker -DockerArguments @('compose', 'exec', '-T', 'oe', 'python3', '/work/tests/ollama_smoke.py', '--report', '/work/.test-results/docker-ollama-smoke.json')
    Write-Host 'Docker Samantha tests passed. Report: work/.test-results/docker-ollama-smoke.json'
} finally {
    Pop-Location
}
