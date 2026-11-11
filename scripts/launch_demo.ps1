param(
    [ValidateSet('train', 'history')][string]$Mode = 'history',
    [ValidateSet('both', 'pendulum', 'car')][string]$Task = 'both',
    [switch]$NoBrowser,
    [int]$AutoClose = -1,
    [int]$Steps = 0
)
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
try {
    $candidates = @()
    if ($env:RLCBF_DESKTOP_PY) { $candidates += $env:RLCBF_DESKTOP_PY }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += $pythonCommand.Source }
    $desktopPython = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $desktopPython = $candidate; break }
    }
    if (-not $desktopPython) {
        $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($pyCommand) {
            try {
                $detected = & $pyCommand.Source -3 -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0) {
                    & $detected -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>$null
                    if ($LASTEXITCODE -eq 0) { $desktopPython = $detected }
                }
            } catch { }
        }
    }
    if (-not $desktopPython) {
        throw 'Python 3.9+ was not found. Install a modern Python with Add to PATH, or set RLCBF_DESKTOP_PY.'
    }
    $launcherArgs = @((Join-Path $PSScriptRoot 'desktop_demo.py'), $Mode, '--task', $Task)
    if ($NoBrowser) { $launcherArgs += '--no-browser' }
    if ($AutoClose -ge 0) { $launcherArgs += @('--auto-close', $AutoClose.ToString()) }
    if ($Steps -gt 0) { $launcherArgs += @('--steps', $Steps.ToString()) }
    & $desktopPython -B @launcherArgs
    exit $LASTEXITCODE
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
