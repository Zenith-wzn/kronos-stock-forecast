param([ValidateSet('preflight','benchmark','run','all')][string]$Stage='all')
$ErrorActionPreference='Stop'
$root=Resolve-Path "$PSScriptRoot/../.."
$env:CUDA_VISIBLE_DEVICES=''; $env:HF_HUB_OFFLINE='1'; $env:PYTHONUNBUFFERED='1'; $env:PYTHONHASHSEED='100'
foreach($model in @('chronos2','timesfm3')) {
    & "$root/$model/.venv/Scripts/python.exe" "$root/scripts/baselines/run_cpu_zero_shot.py" --model $model --stage $Stage
    if($LASTEXITCODE -ne 0) { throw "$model failed with exit code $LASTEXITCODE; next model not started" }
}
