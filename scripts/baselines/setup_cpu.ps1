param([ValidateSet('chronos2','timesfm3')][string]$Model)
$ErrorActionPreference='Stop'
$root=(Resolve-Path "$PSScriptRoot/../..").Path
$env:UV_PYTHON_INSTALL_DIR="$root/.runtime/python"
$env:UV_CACHE_DIR="$root/.runtime/uv-cache"
$specs=@{
 chronos2=@{url='https://github.com/amazon-science/chronos-forecasting.git';commit='4dbf163c2734c089cdf7da2b86fde48862ff9c6f'}
 timesfm3=@{url='https://github.com/google-research/timesfm.git';commit='e31dadd84cb26bd5153fde6687502b8312e918fb'}
}
$homeDir="$root/$Model"
New-Item -ItemType Directory -Force -Path "$homeDir/logs" | Out-Null
if(Test-Path "$homeDir/.venv") { throw 'Environment exists: setup never overwrites an existing environment.' }
if(-not(Test-Path "$homeDir/upstream")) {
 git clone $specs[$Model].url "$homeDir/upstream"
 if($LASTEXITCODE -ne 0){throw 'Clone failed'}
}
git -C "$homeDir/upstream" checkout --detach $specs[$Model].commit
if($LASTEXITCODE -ne 0){throw 'Pinned checkout failed'}
uv python install 3.11.15
if($LASTEXITCODE -ne 0){throw 'Python install failed'}
uv venv "$homeDir/.venv" --python 3.11.15
if($LASTEXITCODE -ne 0){throw 'Environment creation failed'}
$py="$homeDir/.venv/Scripts/python.exe"
uv pip install --python $py 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
if($LASTEXITCODE -ne 0){throw 'CPU torch install failed'}
$lock="$homeDir/requirements.lock.txt"
if(-not(Test-Path $lock)){throw 'Exact dependency lock is missing; do not substitute versions'}
Get-Content $lock | Where-Object {$_ -notmatch '^(torch|chronos-forecasting|timesfm)=='} | Set-Content "$homeDir/dependencies-only.lock.txt"
uv pip install --python $py -r "$homeDir/dependencies-only.lock.txt"
if($LASTEXITCODE -ne 0){throw 'Locked dependency install failed'}
uv pip install --python $py --no-deps "$homeDir/upstream"
if($LASTEXITCODE -ne 0){throw 'Official package install failed'}
$build=Join-Path $homeDir 'upstream/build'
if(Test-Path -LiteralPath $build){
 $src=(Resolve-Path -LiteralPath $build).Path
 $dst=[IO.Path]::GetFullPath((Join-Path $homeDir ('package-build-'+(Get-Date -Format yyyyMMddHHmmss))))
 $base=[IO.Path]::GetFullPath($homeDir)+[IO.Path]::DirectorySeparatorChar
 if(-not $src.StartsWith($base)-or -not $dst.StartsWith($base)){throw 'Unsafe build move'}
 Move-Item -LiteralPath $src -Destination $dst
}
$env:HF_HUB_DISABLE_XET='1'; $env:HF_HUB_DOWNLOAD_TIMEOUT='120'; $env:HF_HOME="$homeDir/.hf-cache"
& $py "$root/scripts/baselines/download_weights.py" --model $Model
if($LASTEXITCODE -ne 0){throw 'Weight download/provenance failed; no alternative checkpoint will be used'}
