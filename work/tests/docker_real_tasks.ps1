param(
    [ValidateSet('Offline', 'Prepare', 'Lab', 'List')]
    [string]$Action = 'Offline',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$taskRepoDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskComposeFile = Join-Path $taskRepoDir 'docker-compose.tests.yaml'
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
    Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'config', '--quiet')
    if (-not $SkipBuild) {
        Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'build', 'tests')
    }
    switch ($Action) {
        'Offline' {
            Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'run', '--rm', '-T', 'tests')
        }
        'Prepare' {
            Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'run', '--rm', '-T', 'task-lab', 'prepare')
        }
        'Lab' {
            Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'run', '--rm', 'task-lab')
        }
        'List' {
            Invoke-TaskDocker -DockerArguments @('compose', '-f', $taskComposeFile, 'run', '--rm', '-T', 'task-lab', 'suite', 'list')
        }
    }
} finally {
    Pop-Location
}
