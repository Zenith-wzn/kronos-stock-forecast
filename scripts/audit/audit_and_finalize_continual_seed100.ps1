[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$project='D:\Desktop\时间序列预测\股票预测'
$root=Join-Path $project 'train3\kronos_csi300_continual_abc_20240701_20250605_seed100'
$run=Join-Path $root 'run'
$python=Join-Path $project '.venv_kronos\Scripts\python.exe'
Write-Output '===ARTIFACT_GATE==='
$summary=Get-Content -LiteralPath (Join-Path $run 'summary.json') -Raw | ConvertFrom-Json
$stderr=Join-Path $root 'logs\runner_formal_seed100.stderr.log'
$files=@('predictions_A.csv','predictions_B.csv','predictions_C.csv','common_predictions.csv','daily_metrics_A.csv','daily_metrics_B.csv','daily_metrics_C.csv','update_log.csv')
$allPresent=$true
foreach($name in $files){$p=Join-Path $run $name;$ok=Test-Path -LiteralPath $p;$allPresent=$allPresent -and $ok;Write-Output "$name=$ok"}
$stderrBytes=(Get-Item -LiteralPath $stderr).Length
Write-Output "summary_status=$($summary.status) common_rows=$($summary.common_rows) stderr_bytes=$stderrBytes all_present=$allPresent"
if($summary.status -ne 'complete' -or -not $allPresent -or $stderrBytes -ne 0){throw 'artifact completion gate failed'}
Write-Output '===METRICS==='
$metricRows=@()
foreach($g in @('A','B','C')){
  $rows=Import-Csv -LiteralPath (Join-Path $run "daily_metrics_$g.csv")
  $rank=@($rows | ForEach-Object {[double]$_.rankic})
  $pear=@($rows | ForEach-Object {[double]$_.pearson_ic})
  $q=@($rows | ForEach-Object {[double]$_.q5_q1})
  $metricRows += [pscustomobject]@{Group=$g;Days=$rows.Count;MeanRankIC=($rank|Measure-Object -Average).Average;PositiveRankICRate=(@($rank|Where-Object {$_ -gt 0}).Count/$rank.Count);MeanPearsonIC=($pear|Measure-Object -Average).Average;MeanQ5Q1=($q|Measure-Object -Average).Average}
}
$metricRows | Format-Table -AutoSize
Write-Output '===BC_DIFFERENCE==='
$b=Import-Csv -LiteralPath (Join-Path $run 'daily_metrics_B.csv');$c=Import-Csv -LiteralPath (Join-Path $run 'daily_metrics_C.csv');$diffs=for($i=0;$i -lt $b.Count;$i++){[double]$c[$i].rankic-[double]$b[$i].rankic};[pscustomobject]@{Days=$diffs.Count;MeanDiff=($diffs|Measure-Object -Average).Average;MaxAbsDiff=($diffs|ForEach-Object {[math]::Abs($_)}|Measure-Object -Maximum).Maximum;NonzeroDays=@($diffs|Where-Object {[math]::Abs($_)-gt 1e-12}).Count}|Format-List
Write-Output '===UPDATE_LOG==='
Import-Csv -LiteralPath (Join-Path $run 'update_log.csv') | Select-Object update_date,model_version_before,model_version_after,champion_validation_rankic_before,candidate_validation_rankic,accepted,decision,selected_replay_rows,selected_recent_rows,selected_historical_rows,next_test_date | Format-Table -AutoSize
Write-Output '===GENERATE_REPORT==='
$report=Join-Path $root 'continual_learning_summary.html'
& $python (Join-Path $project 'generate_continual_csi300_summary_html.py') --run-dir $run --output $report
if($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $report)){throw 'report generation failed'}
$reportItem=Get-Item -LiteralPath $report
@{status='complete';report=$reportItem.FullName;bytes=$reportItem.Length;timestamp=(Get-Date).ToString('o')}|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $root 'REPORT_COMPLETE.json') -Encoding UTF8
Write-Output "REPORT=$($reportItem.FullName) BYTES=$($reportItem.Length)"
Write-Output '===ARCHIVE_FALSE_FAILURE==='
$failedPath=Join-Path $root 'FAILED.json'
if(Test-Path -LiteralPath $failedPath){
 $failed=Get-Content -LiteralPath $failedPath -Raw|ConvertFrom-Json
 if($failed.stage -eq 'runner_formal_seed100' -and $null -eq $failed.exit_code -and $summary.status -eq 'complete' -and $stderrBytes -eq 0){
   $archive=Join-Path $root ('FAILED_FALSE_POSITIVE_RUNNER_EXITCODE_NULL_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.json')
   $resolvedRoot=[IO.Path]::GetFullPath($root);$src=[IO.Path]::GetFullPath($failedPath);$dst=[IO.Path]::GetFullPath($archive)
   if(-not $src.StartsWith($resolvedRoot,[StringComparison]::OrdinalIgnoreCase)-or -not $dst.StartsWith($resolvedRoot,[StringComparison]::OrdinalIgnoreCase)){throw 'path safety check failed'}
   Move-Item -LiteralPath $src -Destination $dst
   Write-Output "ARCHIVED=$archive"
 } else {Write-Output 'FAILED_JSON_NOT_ARCHIVED_UNEXPECTED_CONTENT'}
}else{Write-Output 'NO_FAILED_JSON'}
Write-Output '===FINAL_STATE==='
[pscustomobject]@{Summary=(Test-Path (Join-Path $run 'summary.json'));Report=(Test-Path $report);ReportComplete=(Test-Path (Join-Path $root 'REPORT_COMPLETE.json'));Failed=(Test-Path (Join-Path $root 'FAILED.json'))}|Format-List
