$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path "results\models\final_model.joblib")) {
    Write-Host "Model not found. Run: python scripts\05_tune_models.py"
    exit 1
}
python demo\app.py
