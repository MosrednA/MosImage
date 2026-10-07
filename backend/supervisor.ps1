param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true)][string]$BackendPath
)

$ErrorActionPreference = "Stop"
$env:MOSIMAGE_SUPERVISED = "true"

while ($true) {
    $worker = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $BackendPath `
        -NoNewWindow `
        -PassThru

    $worker.WaitForExit()
    if ($worker.ExitCode -eq 0) {
        exit 0
    }

    Write-Warning "MosImage backend exited with code $($worker.ExitCode). Restarting..."
    Start-Sleep -Seconds 1
}
