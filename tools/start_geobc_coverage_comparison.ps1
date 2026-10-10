[CmdletBinding()]
param([switch]$Start, [string]$Plan = 'notes/geobc_coverage_comparison_plan.json')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$planPath = if ([IO.Path]::IsPathRooted($Plan)) { [IO.Path]::GetFullPath($Plan) } else { [IO.Path]::GetFullPath((Join-Path $workspace $Plan)) }
if (-not $planPath.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Plan must be inside this workspace.' }
$record = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$relativePlan = $planPath.Substring($workspace.Length + 1).Replace('\','/')
if ($relativePlan -notmatch '^[a-zA-Z0-9_./-]+$') { throw 'Plan path contains unsupported command argument characters.' }
$output = Join-Path $workspace $record.training_output
if (-not $Start) {
    [pscustomobject]@{status='source-only-ready';workspace=$workspace;plan=$planPath;queue='tools/run_geobc_coverage_comparison.py';writes=$false} | ConvertTo-Json
    exit 0
}
[IO.Directory]::CreateDirectory($output) | Out-Null
$tag = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')
$stdout = Join-Path $output ('launcher_'+$tag+'.stdout.log')
$stderr = Join-Path $output ('launcher_'+$tag+'.stderr.log')
$argv = '-d cold-surge-ubuntu-24.04 --cd "/mnt/c/Users/ASUS/Desktop/GeoBC Neural Operator" --exec /home/research/.venvs/geobc/bin/python -u tools/run_geobc_coverage_comparison.py --plan '+$relativePlan+' --start'
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $argv -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$receipt = [ordered]@{status='launched';windows_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');plan=$planPath;stdout=$stdout;stderr=$stderr;coordinator_status=(Join-Path $output 'coordinator_status.json');GPU_authority='Shared Linux process-lifetime lock; real root source/phase marker required by actual stages'}
[IO.File]::WriteAllText((Join-Path $output 'launcher.json'),($receipt | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 5
