<#
.SYNOPSIS
Run dynamic Kr upscaling one replay row at a time with a process timeout.

.DESCRIPTION
This supervisor is an operational wrapper around
run_kr_upscaling_dyn_median_examples_full87.m. It does not change the
upscaling algorithm. Instead, each requested curve is launched in its own
MATLAB process so a rare MRST solver stall can be killed without losing all
completed curve checkpoints.

Typical smoke test:
  $env:KR_DYN_SUPERVISED_ROWS = "1"
  $env:KR_DYN_CURVE_TIMEOUT_SECONDS = "600"
  $env:KR_DYN_OUTPUT_TAG = "supervised_smoke"
  $env:KR_DYN_TIMESTEP_MODE = "smoke"
  $env:KR_DYN_COREY_STEP = "3"
  $env:KR_DYN_SMOKE_CARTDIMS = "20,4,20"
  .\run_kr_upscaling_dyn_supervised_full87.ps1

Typical production:
  $env:KR_DYN_CURVE_TIMEOUT_SECONDS = "7200"
  $env:KR_DYN_1D_METHOD = "transport"
  .\run_kr_upscaling_dyn_supervised_full87.ps1 -BuildAggregate
#>

param(
    [string]$Rows = $env:KR_DYN_SUPERVISED_ROWS,
    [int]$TimeoutSeconds = $(if ($env:KR_DYN_CURVE_TIMEOUT_SECONDS) { [int]$env:KR_DYN_CURVE_TIMEOUT_SECONDS } else { 7200 }),
    [int]$FallbackTimeoutSeconds = $(if ($env:KR_DYN_FALLBACK_TIMEOUT_SECONDS) { [int]$env:KR_DYN_FALLBACK_TIMEOUT_SECONDS } else { 7200 }),
    [string]$FallbackOneDAdSolver = $(if ($env:KR_DYN_FALLBACK_1D_AD_SOLVER) { $env:KR_DYN_FALLBACK_1D_AD_SOLVER } else { "robust" }),
    [string]$FallbackOneDMethod = $env:KR_DYN_FALLBACK_1D_METHOD,
    [string]$FallbackTimestepMode = $env:KR_DYN_FALLBACK_TIMESTEP_MODE,
    [string]$MatlabCommand = $(if ($env:MATLAB_COMMAND) { $env:MATLAB_COMMAND } else { "matlab" }),
    [switch]$BuildAggregate
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$sourceRoot = Join-Path "D:\codex_gom\UQ_workflow" "pc_upscaling_median_examples_full87"
$replaySummaryCsv = Join-Path $sourceRoot "tables\replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv"

if (-not (Test-Path -LiteralPath $replaySummaryCsv)) {
    throw "Replay summary not found: $replaySummaryCsv"
}

function Parse-RowList {
    param([string]$Text, [int]$DefaultCount)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return 1..$DefaultCount
    }
    $items = New-Object System.Collections.Generic.List[int]
    foreach ($part in ($Text -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($part)) { continue }
        if ($part -match "^(\d+)-(\d+)$") {
            $a = [int]$Matches[1]
            $b = [int]$Matches[2]
            if ($a -gt $b) { throw "Invalid row range: $part" }
            foreach ($r in $a..$b) { [void]$items.Add($r) }
        } else {
            [void]$items.Add([int]$part)
        }
    }
    return $items.ToArray()
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return }
    & taskkill.exe /PID $ProcessId /T /F | Out-Null
}

function New-MatlabCurveStartInfo {
    param(
        [int]$RowId,
        [string]$DiaryFile,
        [hashtable]$Overrides
    )
    $escapedScriptDir = $scriptDir.Replace("'", "''")
    $escapedDiaryFile = $DiaryFile.Replace("'", "''")
    $matlabBatch = "cd('$escapedScriptDir'); diary('$escapedDiaryFile'); run_kr_upscaling_dyn_median_examples_full87; diary off;"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $MatlabCommand
    $psi.Arguments = "-batch `"$matlabBatch`""
    $psi.WorkingDirectory = $scriptDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    foreach ($key in [System.Environment]::GetEnvironmentVariables("Process").Keys) {
        $psi.Environment[$key] = [System.Environment]::GetEnvironmentVariable($key, "Process")
    }
    $psi.Environment["KR_DYN_ONLY_ROWS"] = [string]$RowId
    $psi.Environment["KR_DYN_USE_PARALLEL"] = "0"
    $psi.Environment["KR_DYN_DISABLE_RUN_MAT_CACHE"] = "1"
    foreach ($key in $Overrides.Keys) {
        if ([string]::IsNullOrWhiteSpace([string]$Overrides[$key])) {
            if ($psi.Environment.ContainsKey($key)) {
                $psi.Environment.Remove($key) | Out-Null
            }
        } else {
            $psi.Environment[$key] = [string]$Overrides[$key]
        }
    }
    return $psi
}

$allRows = Import-Csv -LiteralPath $replaySummaryCsv |
    Where-Object {
        $_.GeologyId -eq "s05_c012" -and
        @("1", "3", "4", "7") -contains $_.Level3CaseId
    }
$rowCount = $allRows.Count
$rowIds = Parse-RowList -Text $Rows -DefaultCount $rowCount
foreach ($rowId in $rowIds) {
    if ($rowId -lt 1 -or $rowId -gt $rowCount) {
        throw "Requested row $rowId is outside 1:$rowCount"
    }
}

$outputTag = if ($env:KR_DYN_OUTPUT_TAG) { $env:KR_DYN_OUTPUT_TAG } else { "supervised" }
$statusRoot = Join-Path "D:\codex_gom\UQ_workflow" "kr_upscaling_dyn_supervised_status_$outputTag"
$logDir = Join-Path $statusRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusCsv = Join-Path $statusRoot "supervised_curve_status.csv"
if (-not (Test-Path -LiteralPath $statusCsv)) {
    "RowId,Status,ExitCode,StartedAt,EndedAt,ElapsedSeconds,DiaryFile" |
        Set-Content -LiteralPath $statusCsv
}

Write-Host "Supervised dynamic Kr rows: $($rowIds -join ', ')"
Write-Host "Timeout per row: $TimeoutSeconds seconds"
if (-not [string]::IsNullOrWhiteSpace($FallbackOneDAdSolver) -or
        -not [string]::IsNullOrWhiteSpace($FallbackOneDMethod) -or
        -not [string]::IsNullOrWhiteSpace($FallbackTimestepMode)) {
    Write-Host "Fallback on timeout: method='$FallbackOneDMethod', adSolver='$FallbackOneDAdSolver', timestep='$FallbackTimestepMode'"
}
Write-Host "Status CSV: $statusCsv"

foreach ($rowId in $rowIds) {
    $started = Get-Date
    $diaryFile = Join-Path $logDir ("row_{0:D5}.log" -f $rowId)

    Write-Host ("Row {0}/{1}: launching MATLAB" -f $rowId, $rowCount)
    $psi = New-MatlabCurveStartInfo -RowId $rowId -DiaryFile $diaryFile -Overrides @{}
    $proc = [System.Diagnostics.Process]::Start($psi)
    $completed = $proc.WaitForExit($TimeoutSeconds * 1000)
    if ($completed) {
        $status = if ($proc.ExitCode -eq 0) { "completed" } else { "failed" }
        $exitCode = $proc.ExitCode
    } else {
        $status = "timed_out"
        $exitCode = -999
        Stop-ProcessTree -ProcessId $proc.Id
    }

    if ($status -eq "timed_out" -and
            (-not [string]::IsNullOrWhiteSpace($FallbackOneDAdSolver) -or
             -not [string]::IsNullOrWhiteSpace($FallbackOneDMethod) -or
             -not [string]::IsNullOrWhiteSpace($FallbackTimestepMode))) {
        $fallbackDiaryFile = Join-Path $logDir ("row_{0:D5}_fallback.log" -f $rowId)
        $fallbackOverrides = @{}
        if (-not [string]::IsNullOrWhiteSpace($FallbackOneDAdSolver)) {
            $fallbackOverrides["KR_DYN_1D_AD_SOLVER"] = $FallbackOneDAdSolver
        }
        if (-not [string]::IsNullOrWhiteSpace($FallbackOneDMethod)) {
            $fallbackOverrides["KR_DYN_1D_METHOD"] = $FallbackOneDMethod
        }
        if (-not [string]::IsNullOrWhiteSpace($FallbackTimestepMode)) {
            $fallbackOverrides["KR_DYN_TIMESTEP_MODE"] = $FallbackTimestepMode
        }
        Write-Host ("Row {0}: primary attempt timed out; launching fallback" -f $rowId)
        $fallbackPsi = New-MatlabCurveStartInfo -RowId $rowId -DiaryFile $fallbackDiaryFile -Overrides $fallbackOverrides
        $fallbackProc = [System.Diagnostics.Process]::Start($fallbackPsi)
        $fallbackCompleted = $fallbackProc.WaitForExit($FallbackTimeoutSeconds * 1000)
        if ($fallbackCompleted) {
            $status = if ($fallbackProc.ExitCode -eq 0) { "fallback_completed" } else { "fallback_failed" }
            $exitCode = $fallbackProc.ExitCode
            $diaryFile = $fallbackDiaryFile
        } else {
            $status = "fallback_timed_out"
            $exitCode = -998
            Stop-ProcessTree -ProcessId $fallbackProc.Id
            $diaryFile = $fallbackDiaryFile
        }
    }
    $ended = Get-Date
    $elapsed = ($ended - $started).TotalSeconds
    """$rowId"",""$status"",""$exitCode"",""$($started.ToString('s'))"",""$($ended.ToString('s'))"",""$([math]::Round($elapsed, 3))"",""$diaryFile""" |
        Add-Content -LiteralPath $statusCsv
    Write-Host ("Row {0}: {1} in {2:n1} seconds" -f $rowId, $status, $elapsed)
}

if ($BuildAggregate) {
    Write-Host "Building aggregate tables from completed curve checkpoints..."
    $escapedScriptDir = $scriptDir.Replace("'", "''")
    $matlabBatch = "cd('$escapedScriptDir'); setenv('KR_DYN_ONLY_ROWS',''); setenv('KR_DYN_DISABLE_RUN_MAT_CACHE','1'); setenv('KR_DYN_USE_PARALLEL','0'); run_kr_upscaling_dyn_median_examples_full87;"
    & $MatlabCommand -batch $matlabBatch
}
