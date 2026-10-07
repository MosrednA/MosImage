param(
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$venvPython = Join-Path $backend ".venv\Scripts\python.exe"
$backendReady = Join-Path $backend ".venv\.mosimage-ready"
$backendSupervisor = Join-Path $backend "supervisor.ps1"

Set-Location $root

function Get-SetupPython {
    $supportedVersions = @("3.13", "3.12")
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($version in $supportedVersions) {
            & py "-$version" -c "import sys; print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = "py"; Args = @("-$version"); Label = "Python $version" }
            }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction Stop
    $versionText = & $pythonCmd.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($supportedVersions -contains $versionText) {
        return @{ Command = $pythonCmd.Source; Args = @(); Label = "Python $versionText" }
    }
    throw "Found Python $versionText, but MosImage requires Python 3.12 or 3.13 for the RTX 5070 Ti CUDA PyTorch setup. Install Python 3.13 or 3.12, then run: Remove-Item backend\.venv -Recurse -Force; .\start.cmd"
}

function Test-CudaTorch {
    param([string]$PythonPath)
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $checkScript = @"
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(2)

import torch
sys.exit(0 if torch.cuda.is_available() else 1)
"@
    try {
        & $PythonPath -c $checkScript *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Get-PythonVersion {
    param([string]$PythonPath)
    return (& $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }

    # Parent PIDs recorded by long-lived Windows processes can be reused later.
    # A real child cannot predate its parent, so exclude stale PID relationships.
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue |
        Where-Object { $_.CreationDate -ge $process.CreationDate }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }
}

if (-not (Test-Path (Join-Path $backend ".env")) -and (Test-Path (Join-Path $backend ".env.example"))) {
    Copy-Item (Join-Path $backend ".env.example") (Join-Path $backend ".env")
    Write-Host "Created backend\.env from backend\.env.example"
}

if (-not (Test-Path $venvPython)) {
    if ($NoInstall) {
        $python = (Get-Command python -ErrorAction Stop).Source
        Write-Host "Using system Python because -NoInstall was provided."
    } else {
        $setupPython = Get-SetupPython
        Write-Host "Creating backend virtual environment with $($setupPython.Label)..."
        & $setupPython.Command @($setupPython.Args + @("-m", "venv", (Join-Path $backend ".venv")))
    }
}

if (Test-Path $venvPython) {
    $python = $venvPython
}

$needsBackendInstall = -not (Test-Path $backendReady)
if (-not $NoInstall -and (Test-Path $venvPython)) {
    $venvVersion = Get-PythonVersion $python
    if (@("3.12", "3.13") -notcontains $venvVersion) {
        throw "backend\.venv uses Python $venvVersion. Delete it and recreate with Python 3.12 or 3.13: Remove-Item backend\.venv -Recurse -Force; .\start.cmd"
    }
    if (-not (Test-CudaTorch $python)) {
        Write-Warning "PyTorch CUDA is not available in backend\.venv. Reinstalling CUDA PyTorch..."
        $needsBackendInstall = $true
    }
}

if (-not $NoInstall -and $needsBackendInstall) {
    Write-Host "Installing backend dependencies. This can take a while on the first run..."
    Push-Location $backend
    & $python -m pip install --upgrade pip
    Write-Host "Installing CUDA PyTorch for RTX 5070 Ti..."
    & $python -m pip install --upgrade --force-reinstall -r requirements-torch-cu130.txt
    & $python -m pip install -r requirements.txt
    & $python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda)"
    New-Item -ItemType File -Path $backendReady -Force | Out-Null
    Pop-Location
}

if (-not $NoInstall -and -not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $frontend
    npm install
    Pop-Location
}

Write-Host "Starting MosImage backend on http://127.0.0.1:8000"
$backendProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $backendSupervisor, "-PythonPath", $python, "-BackendPath", $backend) `
    -WorkingDirectory $root `
    -NoNewWindow `
    -PassThru

try {
    Start-Sleep -Seconds 2
    Write-Host "Starting MosImage frontend on http://localhost:5173"
    Push-Location $frontend
    npm run dev
} finally {
    Pop-Location -ErrorAction SilentlyContinue
    Write-Host "Stopping MosImage backend and worker processes..."
    Stop-ProcessTree -ProcessId $backendProcess.Id
}
