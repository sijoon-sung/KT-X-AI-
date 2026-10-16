$ErrorActionPreference = "Stop"
$SizeRunRoot = Join-Path "runs" ("size-compare-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
$ShareResultRoot = Join-Path "results" (Split-Path $SizeRunRoot -Leaf)
$SizeRows = @()

foreach ($ModelSize in @("base", "medium")) {
    Write-Host "`n========================================================"
    Write-Host " Running Profile: $ModelSize on CUDA"
    Write-Host "========================================================"
    $SizeRun = Join-Path $SizeRunRoot $ModelSize
    & .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile $ModelSize --out $SizeRun --days 35 --stations 12 --epochs 15 --samples 32 --ode-steps 8 --seed 42 --budget 20 --truck-capacity 5 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "$ModelSize execution failed" }
    
    $SizeMeta = Get-Content (Join-Path $SizeRun "run/metadata.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $SizeEval = Get-Content (Join-Path $SizeRun "run/evaluation.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $ShareModelRoot = Join-Path $ShareResultRoot $ModelSize
    New-Item -ItemType Directory -Path $ShareModelRoot -Force | Out-Null
    Copy-Item -LiteralPath @((Join-Path $SizeRun "run/metadata.json"), (Join-Path $SizeRun "run/evaluation.json"), (Join-Path $SizeRun "run/models/training.json")) -Destination $ShareModelRoot
    
    foreach ($ForecastModel in @("xgboost", "gcn", "gcn_flow")) {
        $ValidationResult = $SizeEval.models.$ForecastModel.validation
        $SizeRows += [PSCustomObject]@{
            Size = $ModelSize
            Model = $ForecastModel
            Width = $SizeMeta.hidden
            MAE = $ValidationResult.counts.mae
            CRPS = $ValidationResult.counts.crps
            StockBrier = $ValidationResult.stock.brier
            InventoryMAE = $ValidationResult.stock.net_inventory_mae
            InferenceMs = $ValidationResult.inference.ms_per_city_origin
        }
    }
}

Write-Host "`n========================================================"
Write-Host " Size & Model Comparison Results"
Write-Host "========================================================"
$SizeRows | Format-Table -AutoSize
$SizeRows | Export-Csv (Join-Path $SizeRunRoot "validation_comparison.csv") -NoTypeInformation -Encoding UTF8
Write-Host "Saved comparison to: $SizeRunRoot/validation_comparison.csv"
Copy-Item -LiteralPath (Join-Path $SizeRunRoot "validation_comparison.csv") -Destination $ShareResultRoot
Write-Host "Shareable summaries (no weights or source data): $ShareResultRoot"
