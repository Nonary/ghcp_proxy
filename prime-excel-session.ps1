param(
    [int]$DevToolsPort = 9222,
    [string]$ProxyUrl = "http://127.0.0.1:8000/api/config/excel-session",
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "tools\prime-excel-session.js"
& node $scriptPath `
    --devtools-port $DevToolsPort `
    --proxy-url $ProxyUrl `
    --timeout-ms ($TimeoutSeconds * 1000)
if ($LASTEXITCODE -ne 0) {
    throw "Excel session primer failed with exit code $LASTEXITCODE."
}
