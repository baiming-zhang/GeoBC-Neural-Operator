[CmdletBinding()]
param([switch]$Publish)

# Default is a read-only plan. No credential material is read until -Publish
# has passed completion gates and prepared the isolated local commit.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$rootPrefix = $root + '\'
$receiptPath = Join-Path $root 'notes/github_publish_receipt.json'
$apiVersion = '2026-03-10'
$gitExecutable = (Get-Command git -ErrorAction Stop).Source
$problems = [Collections.Generic.List[string]]::new()
$pruned = [Collections.Generic.List[string]]::new()

function Field($Object, [string]$Name, $Default = $null) {
    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Project-Path([string]$Relative) {
    if ([string]::IsNullOrWhiteSpace($Relative) -or [IO.Path]::IsPathRooted($Relative) -or
        $Relative -eq '.' -or [Management.Automation.WildcardPattern]::ContainsWildcardCharacters($Relative)) {
        throw 'Publication paths must be explicit relative file/directory paths.'
    }
    $absolute = [IO.Path]::GetFullPath((Join-Path $root $Relative))
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'A publication path escapes the project workspace.'
    }
    return $absolute
}

function Native-Quote([string]$Value) {
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Git-Run([string]$Directory, [string[]]$Arguments, [string]$InputText = $null,
                 [int[]]$AcceptedExitCodes = @(0)) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $gitExecutable
    $allArguments = @('-C', $Directory) + $Arguments
    $info.Arguments = (($allArguments | ForEach-Object { Native-Quote $_ }) -join ' ')
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Text.UTF8Encoding]::new($false)
    $info.StandardErrorEncoding = [Text.UTF8Encoding]::new($false)
    foreach ($name in @('GIT_TRACE', 'GIT_TRACE_CURL', 'GIT_CURL_VERBOSE', 'GCM_TRACE', 'GCM_TRACE_SECRETS')) {
        $info.EnvironmentVariables[$name] = '0'
    }
    $info.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'
    $info.EnvironmentVariables['GCM_INTERACTIVE'] = 'never'
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        if ($null -ne $InputText) { $process.StandardInput.Write($InputText) }
        $process.StandardInput.Close()
        $outputTask = $process.StandardOutput.ReadToEndAsync()
        $errorTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $output = $outputTask.Result
        # Do not expose stderr: helpers or HTTP diagnostics can contain credentials.
        [void]$errorTask.Result
        if ($AcceptedExitCodes -notcontains $process.ExitCode) {
            throw ('Git operation failed: {0}; exit {1}. Diagnostic output withheld.' -f $Arguments[0], $process.ExitCode)
        }
        return [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $output }
    } finally { $process.Dispose() }
}

function Read-Json([string]$Path) {
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Save-Receipt($Receipt) {
    $Receipt.updated_at_utc = [DateTime]::UtcNow.ToString('o')
    $json = $Receipt | ConvertTo-Json -Depth 16
    $temporary = $receiptPath + '.tmp'
    [IO.File]::WriteAllText($temporary, $json + "`n", [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
    if (Test-Path -LiteralPath $snapshot -PathType Container) {
        $snapshotReceipt = Join-Path $snapshot 'notes/github_publish_receipt.json'
        [IO.Directory]::CreateDirectory((Split-Path -Parent $snapshotReceipt)) | Out-Null
        [IO.File]::WriteAllText($snapshotReceipt, $json + "`n", [Text.UTF8Encoding]::new($false))
    }
}

function Inventory([string[]]$Allowlist) {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $files = [Collections.Generic.List[object]]::new()
    $skipNames = @('.git', '__pycache__', '.venv', 'venv', '.local_deps', 'cache', '.cache',
                   '.pytest_cache', '.mypy_cache', '.ruff_cache', '.tox', 'node_modules')
    foreach ($relative in $Allowlist) {
        $absolute = Project-Path $relative
        if (-not (Test-Path -LiteralPath $absolute)) { $problems.Add('Missing allowlist path: ' + $relative); continue }
        $pending = [Collections.Generic.Stack[object]]::new()
        $pending.Push((Get-Item -LiteralPath $absolute -Force))
        while ($pending.Count) {
            $item = $pending.Pop()
            if ($skipNames -contains $item.Name -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                $pruned.Add($item.FullName.Substring($rootPrefix.Length).Replace('\', '/')); continue
            }
            if ($item.PSIsContainer) {
                foreach ($child in Get-ChildItem -LiteralPath $item.FullName -Force) { $pending.Push($child) }
            } elseif ($item.Extension -notin @('.pyc', '.pyo')) {
                $name = $item.FullName.Substring($rootPrefix.Length).Replace('\', '/')
                if ($name -eq 'notes/github_publish_receipt.json') { continue }
                if ($seen.Add($name)) {
                    if ($item.Length -gt 95MB) { $problems.Add('File exceeds 95 MiB: ' + $name) }
                    $files.Add([pscustomobject]@{ Relative = $name; Source = $item.FullName; Bytes = $item.Length })
                }
            }
        }
    }
    return $files | Sort-Object Relative
}

function Api([string]$Method, [string]$Endpoint, $Body = $null, [switch]$Allow404) {
    $parameters = @{ Method = $Method; Uri = 'https://api.github.com' + $Endpoint;
                     Headers = $script:apiHeaders; TimeoutSec = 60; ErrorAction = 'Stop' }
    if ($null -ne $Body) { $parameters.Body = ($Body | ConvertTo-Json -Compress); $parameters.ContentType = 'application/json' }
    try { return Invoke-RestMethod @parameters } catch {
        $status = 0
        if ($null -ne $_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
        if ($Allow404 -and $status -eq 404) { return $null }
        throw ('GitHub REST {0} {1} failed (HTTP {2}); credentials and response details withheld.' -f $Method, $Endpoint, $status)
    }
}

function Assert-Repository($Repository, [long]$UserId) {
    if ($null -eq $Repository -or $Repository.private -ne $true -or $Repository.owner.id -ne $UserId -or
        $Repository.owner.login -ine $plan.owner -or $Repository.name -ine $plan.name -or
        $Repository.full_name -ine ($plan.owner + '/' + $plan.name)) {
        throw 'Repository identity/owner/private verification failed; no push permitted.'
    }
}

$planPath = Join-Path $root 'notes/github_publish_plan.json'
if (-not (Test-Path -LiteralPath $planPath)) {
    if ($Publish) { throw 'Publication plan is missing; nothing was changed.' }
    [pscustomobject]@{ mode = 'dry-run'; ready = $false; reasons = @('Publication plan is missing'); writes = $false; credentials_read = $false } | ConvertTo-Json
    exit 0
}
$plan = Read-Json $planPath
if ($plan.owner -cne 'baiming-zhang' -or $plan.name -cne 'GeoBC-Neural-Operator' -or
    $plan.private -isnot [bool] -or -not $plan.private -or $plan.github_user_id -ne 174442095) {
    throw 'Publication plan differs from the authorized private repository/account.'
}
$snapshotRelative = Field $plan 'checkout_directory' 'publication/GeoBC-Neural-Operator'
$snapshot = Project-Path $snapshotRelative
if (-not $snapshot.StartsWith((Join-Path $root 'publication') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The publication checkout must be isolated under publication/.'
}
$jobs = @((Read-Json (Join-Path $root 'notes/all_methods_plan.json')).jobs)
if ($jobs.Count -ne 30) { $problems.Add('Expected exactly 30 suite experiments.') }
$completed = 0
foreach ($job in $jobs) {
    $summaryPath = Join-Path (Project-Path $job.output) 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath)) { $problems.Add('Summary missing: ' + $job.name); continue }
    try {
        if ((Field (Read-Json $summaryPath) 'status') -eq 'completed') { $completed++ }
        else { $problems.Add('Experiment is not completed: ' + $job.name) }
    } catch { $problems.Add('Summary is not readable: ' + $job.name) }
}
$manifestRelative = Field $plan 'results_manifest' 'results/manifest.json'
$manifestPath = Project-Path $manifestRelative
$manifestComplete = $false
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Read-Json $manifestPath
    $manifestComplete = ((Field $manifest 'complete') -eq $true -or (Field $manifest 'status') -in @('complete', 'completed'))
}
if (-not $manifestComplete) { $problems.Add('Final results manifest is missing or incomplete.') }
$files = @(Inventory @($plan.allowlist))
$accounts = (Git-Run $root @('credential-manager', 'github', 'list')).Output -split '\r?\n'
$accountSaved = @($accounts | Where-Object { $_.Trim() -ieq $plan.owner }).Count -gt 0
if (-not $accountSaved) { $problems.Add('Expected Git Credential Manager account is not saved.') }
$origin = 'https://github.com/' + $plan.owner + '/' + $plan.name + '.git'
$totalBytes = ($files | Measure-Object Bytes -Sum).Sum
$preview = [ordered]@{ mode = $(if ($Publish) { 'publish' } else { 'dry-run' }); ready = ($problems.Count -eq 0);
    repository = $plan.owner + '/' + $plan.name; private = $true; completed = $completed; expected = 30;
    results_manifest_complete = $manifestComplete; gcm_expected_account_saved = $accountSaved;
    snapshot = $snapshotRelative; snapshot_file_count = $files.Count; snapshot_bytes = $totalBytes;
    pruned_git_cache_or_link_paths = @($pruned); blocking_reasons = @($problems);
    local_commit_before_repository_creation = $true; credentials_read = $false; writes = $false }
$preview | ConvertTo-Json -Depth 5
if (-not $Publish) { exit 0 }
if ($problems.Count) { throw 'Publication completion/content gates have not passed; no credentials or repository were accessed.' }

$receipt = $null
if (Test-Path -LiteralPath $receiptPath) {
    $receipt = Read-Json $receiptPath
    if ($receipt.project_id -ne $plan.project_id -or $receipt.owner -ine $plan.owner -or $receipt.name -ine $plan.name -or
        $receipt.checkout_directory -ne $snapshotRelative) { throw 'Existing receipt does not belong to this project/checkout.' }
}
if (Test-Path -LiteralPath $snapshot) {
    if ($null -eq $receipt -and @(Get-ChildItem -LiteralPath $snapshot -Force).Count) {
        throw 'An unrecognized publication checkout already exists; it will not be overwritten.'
    }
} else { [IO.Directory]::CreateDirectory($snapshot) | Out-Null }
foreach ($file in $files) {
    $destination = Join-Path $snapshot $file.Relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $file.Source -Destination $destination -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $snapshot '.git'))) { [void](Git-Run $snapshot @('init', '--initial-branch=main')) }
[void](Git-Run $snapshot @('config', '--local', 'user.name', $plan.owner))
[void](Git-Run $snapshot @('config', '--local', 'user.email', '174442095+baiming-zhang@users.noreply.github.com'))
[void](Git-Run $snapshot @('config', '--local', 'core.quotepath', 'false'))
[void](Git-Run $snapshot @('config', '--local', 'credential.https://github.com.username', $plan.owner))
$tracked = (Git-Run $snapshot @('ls-files')).Output -split '\r?\n' | Where-Object { $_ }
$allowedNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($file in $files) { [void]$allowedNames.Add($file.Relative) }
foreach ($name in $tracked) { if (-not $allowedNames.Contains($name)) { throw 'Publication checkout has tracked content outside the current allowlist inventory.' } }
for ($i = 0; $i -lt $files.Count; $i += 50) {
    $last = [Math]::Min($i + 49, $files.Count - 1)
    [string[]]$batch = @($files[$i..$last] | ForEach-Object { $_.Relative })
    [void](Git-Run $snapshot (@('add', '--force', '--') + $batch))
}
$diff = Git-Run $snapshot @('diff', '--cached', '--quiet') -AcceptedExitCodes @(0, 1)
if ($diff.ExitCode -eq 1) { [void](Git-Run $snapshot @('commit', '-m', 'Reproduce original DeepJEB fitting experiments and publish completed results')) }
$commit = (Git-Run $snapshot @('rev-parse', 'HEAD')).Output.Trim()
if ((Git-Run $snapshot @('branch', '--show-current')).Output.Trim() -ne 'main') { throw 'Publication checkout must use main.' }
if ($null -eq $receipt) {
    $receipt = [pscustomobject]@{ project_id = $plan.project_id; owner = $plan.owner; name = $plan.name;
        private = $true; checkout_directory = $snapshotRelative; repository_id = $null;
        creation_attempted = $false; creation_confirmed = $false; phase = 'local-prepared';
        created_at_utc = [DateTime]::UtcNow.ToString('o'); updated_at_utc = ''; local_commit = '';
        origin_url = $origin; snapshot_file_count = $files.Count; snapshot_bytes = $totalBytes;
        plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash;
        repo_url = $null; creation_probe_repository_id = $null; published_at_utc = $null }
}
$receipt.local_commit = $commit
$receipt.phase = 'local-prepared'
Save-Receipt $receipt
Write-Output ('Prepared isolated local commit ' + $commit + '; no remote repository has been changed yet.')

$credential = @{}
$script:apiHeaders = @{}
try {
    $credentialOutput = (Git-Run $root @('credential', 'fill') -InputText ("protocol=https`nhost=github.com`nusername=" + $plan.owner + "`n`n")).Output
    foreach ($line in $credentialOutput -split '\r?\n') {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) { $credential[$parts[0]] = $parts[1] }
    }
    $credentialOutput = $null
    if (-not $credential.ContainsKey('password') -or [string]::IsNullOrWhiteSpace($credential['password'])) { throw 'Git Credential Manager supplied no GitHub credential.' }
    $script:apiHeaders = @{ Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = $apiVersion;
        'User-Agent' = 'GeoBC-Neural-Operator-private-publisher'; Authorization = 'Bearer ' + $credential['password'] }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $user = Api 'GET' '/user'
    if ($user.login -ine $plan.owner -or $user.id -ne $plan.github_user_id) { throw 'Authenticated GitHub account does not match the authorized owner.' }
    $endpoint = '/repos/' + $plan.owner + '/' + $plan.name
    $repository = Api 'GET' $endpoint -Allow404
    if ($null -ne $repository) {
        Assert-Repository $repository $user.id
        if (-not $receipt.creation_confirmed -or $null -eq $receipt.repository_id -or $receipt.repository_id -ne $repository.id) {
            throw 'Same-named repository already exists without this project''s confirmed receipt; it will not be changed.'
        }
    } else {
        if ($receipt.creation_attempted) { throw 'A prior repository creation attempt exists; refusing a duplicate POST. Inspect its receipt and remote state.' }
        $receipt.creation_attempted = $true
        $receipt.phase = 'creation-attempted'
        Save-Receipt $receipt
        try {
            $repository = Api 'POST' '/user/repos' @{ name = $plan.name; private = $true; auto_init = $false; description = $plan.description }
        } catch {
            # Creation timeouts/5xx may have succeeded server-side. Probe once,
            # never retry POST or push an unconfirmed same-named repository.
            try {
                $probe = Api 'GET' $endpoint -Allow404
                if ($null -ne $probe) { $receipt.creation_probe_repository_id = $probe.id }
            } catch { }
            $receipt.phase = 'creation-unconfirmed'
            Save-Receipt $receipt
            throw 'Repository creation was not confirmed. Remote state was probed; no duplicate creation or push was attempted.'
        }
        Assert-Repository $repository $user.id
        $receipt.repository_id = $repository.id
        $receipt.creation_confirmed = $true
        $receipt.repo_url = $repository.html_url
        $receipt.phase = 'repository-created'
        Save-Receipt $receipt
    }
    $remote = Git-Run $snapshot @('remote', 'get-url', 'origin') -AcceptedExitCodes @(0, 2)
    if ($remote.ExitCode -eq 0) {
        if ($remote.Output.Trim() -ine $origin) { throw 'Existing origin differs from the authorized target; no push permitted.' }
    } else { [void](Git-Run $snapshot @('remote', 'add', 'origin', $origin)) }
    Assert-Repository (Api 'GET' $endpoint) $user.id
    [void](Git-Run $snapshot @('push', '--set-upstream', 'origin', 'main'))
    $remoteRef = (Git-Run $snapshot @('ls-remote', 'origin', 'refs/heads/main')).Output.Trim()
    if (-not $remoteRef.StartsWith($commit + "`t")) { throw 'Remote main does not match the prepared local commit.' }
    Assert-Repository (Api 'GET' $endpoint) $user.id
    $receipt.phase = 'published'
    $receipt.published_at_utc = [DateTime]::UtcNow.ToString('o')
    Save-Receipt $receipt
    Write-Output ('Published private repository https://github.com/' + $plan.owner + '/' + $plan.name + ' at ' + $commit)
} catch {
    # Function errors are intentionally static/status-only and never contain
    # raw HTTP/helper diagnostics, headers or credential values.
    $receipt.phase = $(if ($receipt.phase -eq 'creation-unconfirmed') { 'creation-unconfirmed' } else { 'failed' })
    Save-Receipt $receipt
    throw
} finally {
    $credential.Clear()
    $script:apiHeaders.Clear()
    $credentialOutput = $null
}
