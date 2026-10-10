[CmdletBinding()]
param(
    [switch]$Start,
    [string]$Plan = 'notes/geobc_coverage_comparison_plan.json',
    [string]$Python = 'C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$planPath = if ([IO.Path]::IsPathRooted($Plan)) { [IO.Path]::GetFullPath($Plan) } else { [IO.Path]::GetFullPath((Join-Path $projectRoot $Plan)) }
$payload = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($payload.kind -ne 'geobc_solver_coverage40vs200_transolver_all_versions') { throw 'Unexpected coverage plan kind.' }
$output = [IO.Path]::GetFullPath((Join-Path $projectRoot $payload.acquisition.output))
$expectedOutput = [IO.Path]::GetFullPath((Join-Path $projectRoot 'data/raw_geobc_coverage'))
if ($output -ne $expectedOutput) { throw 'Only the frozen isolated output is allowed.' }
if (-not $Start) {
    [pscustomobject]@{status='source-ready';module='tools.fetch_geobc_coverage';plan=$planPath;output=$output;workers=2;network_requests=0;writes=$false} | ConvertTo-Json
    exit 0
}
$reviewPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $payload.root_review.path))
if ($reviewPath -ne [IO.Path]::GetFullPath((Join-Path $projectRoot 'notes/geobc_coverage_root_review.json'))) { throw 'Unexpected root review path.' }
$review = Get-Content -LiteralPath $reviewPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($review.passed -ne $true -or $review.acquisition_released -ne $true -or $review.scientific_digest -ne $payload.scientific_digest) { throw 'Authentic binary acquisition release is missing.' }
# The module verifies all source pins and the complete frozen inventory before
# network. No preparation/training launch or credentials/proxy mutation occurs.
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Configured bundled Windows Python is unavailable.' }
if ($planPath.Contains('"') -or $Python.Contains('"')) { throw 'Invalid quoted native argument.' }
$lockPath = Join-Path $output '.fetch.lock'
if (Test-Path -LiteralPath $lockPath) {
    $savedLock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $priorProcess = Get-Process -Id ([int]$savedLock.pid) -ErrorAction SilentlyContinue
    if ($null -ne $priorProcess) {
        [pscustomobject]@{status='existing-process-requires-lock-check';pid=$savedLock.pid;writes=$false} | ConvertTo-Json
        exit 0
    }
}
[IO.Directory]::CreateDirectory($output) | Out-Null
$logRoot = Join-Path $output '.runtime'
[IO.Directory]::CreateDirectory($logRoot) | Out-Null
$tag = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssffff')
$stdout = Join-Path $logRoot ($tag + '.stdout.log')
$stderr = Join-Path $logRoot ($tag + '.stderr.log')
$arguments = '-u -m tools.fetch_geobc_coverage --plan "' + $planPath + '"'
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$receipt = [ordered]@{status='launched';windows_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');plan=$planPath;scientific_digest=$payload.scientific_digest;output=$output;stdout=$stdout;stderr=$stderr;workers=2;binary_only=$true;labels_sealed=$true;GPU_work=$false}
$receiptPath = Join-Path $logRoot ($tag + '.launch.json')
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 5
