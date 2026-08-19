# Test runner for the BaCP verification ladder.
#
#   .\run_tests.ps1           # fast suite  (~8s, no GPU)  -- use this while editing
#   .\run_tests.ps1 -Full     # everything incl. training loops (~2 min)
#   .\run_tests.ps1 -Level L4 # one level, e.g. L0 L1 L2 L3 L4 L5 L6a L6b L6c L7 L8 L9
#
# First-time setup:
#   python -m venv .venv
#   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

param(
    [switch]$Full,
    [string]$Level = "",
    [switch]$Durations
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Create it with:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt"
    exit 1
}

# Levels map to files. Each level depends only on the levels below it, so when
# several fail at once, fix the lowest one first.
$levelFiles = @{
    "L0"  = "project/tests/test_imports.py"
    "L1"  = "project/tests/test_losses.py"
    "L1b" = "project/tests/test_cap_contrastive.py"
    "L2"  = "project/tests/test_data_invariants.py"
    "L3"  = "project/tests/test_model_forward.py"
    "L4"  = "project/tests/test_pruner_masks.py"
    "L5"  = "project/tests/test_sparsity_accounting.py"
    "L6a" = "project/tests/test_dyrelu_phasing.py"
    "L6b" = "project/tests/test_weight_sharing.py"
    "L6c" = "project/tests/test_east_pruner.py"
    "L7"  = "project/tests/test_training_step.py"
    "L8"  = "project/tests/test_baseline_loop.py"
    "L9"  = "project/tests/test_bacp_loop.py"
}

$args = @("-m", "pytest", "-W", "ignore::DeprecationWarning", "-W", "ignore::FutureWarning")

if ($Level) {
    if (-not $levelFiles.ContainsKey($Level)) {
        Write-Host "Unknown level '$Level'. Known: $($levelFiles.Keys -join ', ')" -ForegroundColor Red
        exit 1
    }
    $args += $levelFiles[$Level]
    Write-Host "Running $Level -> $($levelFiles[$Level])" -ForegroundColor Cyan
}
elseif ($Full) {
    Write-Host "Running the full suite (includes training loops)" -ForegroundColor Cyan
}
else {
    $args += @("-m", "not slow")
    Write-Host "Running the fast suite. Use -Full to include the training loops." -ForegroundColor Cyan
}

if ($Durations) { $args += "--durations=15" }

& $python @args
exit $LASTEXITCODE
