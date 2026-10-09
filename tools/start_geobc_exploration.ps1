[CmdletBinding()]
param([switch]$Start, [string[]]$Only = @())

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$outputRoot = Join-Path $projectRoot 'runs/geobc_exploration'
$coordinatorPath = Join-Path $outputRoot 'coordinator_status.json'
if (Test-Path -LiteralPath $coordinatorPath) {
    $saved = Get-Content -LiteralPath $coordinatorPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($saved.phase -in @('running','training','evaluating','waiting_for_inputs')) {
        # Linux flock is the final authority. This avoids needlessly launching
        # a second Windows helper while an existing coordinator is active.
        $probe = & wsl.exe -d cold-surge-ubuntu-24.04 --exec /usr/bin/python3 -c "import pathlib,sys; p=pathlib.Path('/proc/'+str(int(sys.argv[1]))+'/cmdline'); argv=p.read_bytes().split(bytes([0])) if p.exists() else []; print('alive' if any(x==b'tools/run_geobc_exploration.py' or x.endswith(b'/tools/run_geobc_exploration.py') for x in argv) else 'gone')" $saved.pid
        if ($LASTEXITCODE -eq 0 -and $probe.Trim() -eq 'alive') {
            [pscustomobject]@{status='already-running';pid=$saved.pid;phase=$saved.phase;writes=$false} | ConvertTo-Json
            exit 0
        }
    }
}
if (-not $Start) {
    [pscustomobject]@{status='ready';workspace=$projectRoot;queue='tools/run_geobc_exploration.py';writes=$false} | ConvertTo-Json
    exit 0
}
[IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$arguments = '-d cold-surge-ubuntu-24.04 --cd "/mnt/c/Users/ASUS/Desktop/GeoBC Neural Operator" --exec /home/research/.venvs/geobc/bin/python -u tools/run_geobc_exploration.py'
if ($Only.Count) {
    foreach ($name in $Only) { if ($name -notmatch '^[a-zA-Z0-9_]+$') { throw 'Only contains an invalid job name.' } }
    $arguments += ' --only ' + ($Only -join ' ')
}
$tag = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')
$stdout = Join-Path $outputRoot ('launcher_' + $tag + '.stdout.log')
$stderr = Join-Path $outputRoot ('launcher_' + $tag + '.stderr.log')
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$receipt = [ordered]@{phase='launched';windows_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');only=$Only;stdout=$stdout;stderr=$stderr;coordinator_status=$coordinatorPath}
[IO.File]::WriteAllText((Join-Path $outputRoot 'launcher.json'),($receipt | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 5
