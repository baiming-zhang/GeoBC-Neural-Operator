[CmdletBinding()]
param([switch]$Push,
      [ValidatePattern('^[0-9a-f]{40}$')][string]$TargetCommit = 'f621a3bc5570cc1056d0ba24a4a18e90bf43a0c9')

# SOURCE-ONLY unless -Push is supplied after the parent's source review.
# No copying, staging, committing, fetching, repository creation, or force push.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$snapshot = Join-Path $root 'publication/GeoBC-Neural-Operator'
$target = $TargetCommit.ToLowerInvariant()
$owner = 'baiming-zhang'
$repoName = 'GeoBC-Neural-Operator'
$userId = [long]174442095
$repoId = [long]1410470321
$receiptPath = Join-Path $root 'notes/github_immutable_push_receipt.json'
$reviewPath = Join-Path $root 'notes/github_immutable_push_root_review.json'
$remoteUrl = 'https://github.com/baiming-zhang/GeoBC-Neural-Operator.git'

if (-not $Push) {
    [ordered]@{
        mode = 'source-only'; target_commit = $target; mutation_attempted = $false
        execution_requires_parent_source_review = $true
        route = 'process-only direct HTTP/1.1; unverified'
        prior_inherited_proxy_probe_category = 'proxy_transport'
        prior_remote_main_sha = $null
        snapshot_recopy = $false; new_commit = $false; force_push = $false
    } | ConvertTo-Json
    exit 0
}

if ($PSVersionTable.PSVersion -lt [Version]'7.2') {
    throw 'Actual execution requires PowerShell 7.2+ for process-tree timeouts and HTTP version policy.'
}
$gitExecutable = (Get-Command git -ErrorAction Stop).Source
$watch = [Diagnostics.Stopwatch]::StartNew()
$script:safeFailure = $null
$script:apiHeaders = $null
$script:httpClient = $null
$credentialOutput = $null
$password = $null
$receipt = [ordered]@{
    kind = 'reviewed_immutable_private_snapshot_push'; status = 'running'
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    root_review_sha256 = $null
    target_commit = $target; local_head_before = $null; local_head_after = $null
    expected_identity = @{ owner = $owner; user_id = $userId; repository_id = $repoId; private = $true }
    route = @{ direct = $true; HTTP_version = '1.1'; route_actual_validation_pending = $true
        global_proxy_changes = 0; TLS_policy_changes = 0
        Git_proxy = 'disabled only in child process'; REST_proxy = 'UseProxy=false'
        readonly_hard_timeout_seconds = 45; push_hard_timeout_seconds = 300
        REST_hard_timeout_seconds = 45; low_speed_bytes_per_second = 1000; low_speed_seconds = 30
        connection_timeout_bound = 'each operation hard deadline; no unsupported Git connect-timeout setting' }
    prior_probe = @{ route = 'inherited localhost proxy'; elapsed_seconds = 20.8277243
        exit_code = 128; category = 'proxy_transport'; HTTP_codes = @(); remote_main_sha = $null }
    verification_before = $null; verification_after = $null
    remote_main_before = $null; remote_main_after = $null
    ancestor_check = $null; push_attempted = $false; push_exit_code = $null
    remote_outcome = 'not_observed'; completed = $false
    snapshot_recopy = $false; staging_operations = 0; new_commits = 0
    force_push = $false; fetch_operations = 0; repository_creations = 0
    credentials_written_by_script = $false; raw_stderr_persisted = $false
    credential_helper_storage_policy = 'existing Git/GCM helper; this script does not export credentials'
    operations = [Collections.Generic.List[object]]::new(); failure_category = $null
    elapsed_seconds = $null; finished_at_utc = $null
}

function Fail-Safe([string]$Category) {
    $script:safeFailure = $Category
    throw 'Operation stopped; only the allowlisted failure category is recorded.'
}

function Field($Object, [string]$Name, $Default = $null) {
    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Native-Quote([string]$Value) {
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Classify-Stderr([string]$Raw) {
    $category = 'unspecified'
    if ($Raw -match 'proxy|CONNECT tunnel') { $category = 'proxy_transport' }
    elseif ($Raw -match 'Authentication failed|could not read Username|invalid credentials') { $category = 'authentication' }
    elseif ($Raw -match 'permission.*denied|not authorized|write access.*denied') { $category = 'permission' }
    elseif ($Raw -match 'SSL|schannel|certificate') { $category = 'TLS' }
    elseif ($Raw -match 'Could not resolve|Failed to connect|Couldn.t connect|timed out') { $category = 'network_connection' }
    elseif ($Raw -match 'non-fast-forward|fetch first|rejected') { $category = 'branch_conflict' }
    elseif ($Raw -match 'RPC failed|HTTP/2|returned error: [45][0-9][0-9]') { $category = 'network_HTTP' }
    $codes = @([regex]::Matches($Raw, '(?:HTTP(?:/[0-9.]+)?\s+|returned error:\s*)([45][0-9]{2})') |
        ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique)
    return @{ category = $category; HTTP_codes = $codes }
}

function Git-Call([string]$Operation, [string[]]$Arguments,
                  [int]$TimeoutSeconds = 45, [string]$InputText = $null) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $gitExecutable
    $argsAll = @('-C', $snapshot, '-c', 'http.version=HTTP/1.1', '-c', 'http.proxy=',
        '-c', 'http.lowSpeedLimit=1000', '-c', 'http.lowSpeedTime=30') + $Arguments
    $info.Arguments = (($argsAll | ForEach-Object { Native-Quote $_ }) -join ' ')
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true; $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Text.UTF8Encoding]::new($false)
    $info.StandardErrorEncoding = [Text.UTF8Encoding]::new($false)
    # Only the new child environment changes; parent/global proxy settings stay intact.
    foreach ($name in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy')) {
        [void]$info.Environment.Remove($name)
    }
    foreach ($name in @('GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE','GCM_TRACE','GCM_TRACE_SECRETS')) {
        $info.Environment[$name] = '0'
    }
    $info.Environment['GIT_TERMINAL_PROMPT'] = '0'; $info.Environment['GCM_INTERACTIVE'] = 'never'
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $rawOut = ''; $rawErr = ''; $timedOut = $false; $exitCode = $null
    try {
        [void]$process.Start()
        $outTask = $process.StandardOutput.ReadToEndAsync()
        $errTask = $process.StandardError.ReadToEndAsync()
        if ($null -ne $InputText) { $process.StandardInput.Write($InputText) }
        $process.StandardInput.Close()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $timedOut = $true
            try { $process.Kill($true) } catch { }
            [void]$process.WaitForExit(5000)
        }
        if ($process.HasExited) { $exitCode = $process.ExitCode }
        if (-not $outTask.IsCompleted) { [void]$outTask.Wait(5000) }
        if (-not $errTask.IsCompleted) { [void]$errTask.Wait(5000) }
        if ($outTask.IsCompletedSuccessfully) { $rawOut = $outTask.Result }
        if ($errTask.IsCompletedSuccessfully) { $rawErr = $errTask.Result }
        $safe = Classify-Stderr $rawErr
        if ($timedOut) { $safe.category = 'hard_timeout' }
        $receipt.operations.Add(@{ operation = $Operation; exit_code = $exitCode; timeout = $timedOut
            elapsed_seconds = $timer.Elapsed.TotalSeconds; category = $safe.category; HTTP_codes = $safe.HTTP_codes })
        # Stdout stays private to the caller. It is never printed or saved wholesale.
        return @{ exit_code = $exitCode; timeout = $timedOut; output = $rawOut; category = $safe.category }
    } finally {
        $rawErr = $null; $InputText = $null; $process.Dispose()
    }
}

function Require-Git($Result, [string]$Category) {
    if ($Result.timeout -or $Result.exit_code -ne 0) { Fail-Safe ($Category + ':' + $Result.category) }
    return $Result.output.Trim()
}

function Api-Get([string]$Endpoint, [string]$Operation) {
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, 'https://api.github.com' + $Endpoint)
    $request.Version = [Version]'1.1'
    $request.VersionPolicy = [Net.Http.HttpVersionPolicy]::RequestVersionExact
    foreach ($name in $script:apiHeaders.Keys) { [void]$request.Headers.TryAddWithoutValidation($name, $script:apiHeaders[$name]) }
    $cancel = [Threading.CancellationTokenSource]::new(45000)
    $response = $null; $body = $null; $timer = [Diagnostics.Stopwatch]::StartNew(); $code = 0
    try {
        $response = $script:httpClient.SendAsync($request, $cancel.Token).GetAwaiter().GetResult()
        $code = [int]$response.StatusCode
        if (-not $response.IsSuccessStatusCode) { Fail-Safe ('REST_HTTP_' + $code) }
        $body = $response.Content.ReadAsStringAsync($cancel.Token).GetAwaiter().GetResult()
        return ($body | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        if ($null -eq $script:safeFailure) { $script:safeFailure = 'REST_transport_or_response' }
        throw 'REST operation failed; response, token and raw diagnostics withheld.'
    } finally {
        $receipt.operations.Add(@{ operation = $Operation; HTTP_code = $code
            elapsed_seconds = $timer.Elapsed.TotalSeconds; raw_response_persisted = $false })
        $body = $null; if ($null -ne $response) { $response.Dispose() }; $request.Dispose(); $cancel.Dispose()
    }
}

function Verify-Identity([string]$Phase) {
    $user = Api-Get '/user' ('REST_user_' + $Phase)
    $repo = Api-Get ('/repos/' + $owner + '/' + $repoName) ('REST_repository_' + $Phase)
    $repoOwner = Field $repo 'owner'
    if ((Field $user 'login') -cne $owner -or (Field $user 'id') -ne $userId -or
        (Field $repo 'id') -ne $repoId -or (Field $repo 'private') -isnot [bool] -or -not $repo.private -or
        (Field $repoOwner 'login') -cne $owner -or (Field $repoOwner 'id') -ne $userId -or
        (Field $repo 'name') -cne $repoName -or (Field $repo 'full_name') -cne ($owner + '/' + $repoName)) {
        Fail-Safe 'identity_or_privacy_mismatch'
    }
    return @{ passed = $true; user_id = $userId; owner = $owner; repository_id = $repoId; private = $true }
}

function Read-Remote-Main([string]$Phase) {
    $result = Git-Call ('ls_remote_' + $Phase) @('ls-remote', 'origin', 'refs/heads/main')
    $value = Require-Git $result 'remote_ref_unavailable'
    if ($value -eq '') { return $null }
    if ($value -notmatch '^([0-9a-fA-F]{40})\s+refs/heads/main$') { Fail-Safe 'unexpected_remote_ref_shape' }
    return $Matches[1].ToLowerInvariant()
}

try {
    if (-not (Test-Path -LiteralPath $reviewPath -PathType Leaf)) { Fail-Safe 'root_source_and_target_review_missing' }
    $approval = [IO.File]::ReadAllText($reviewPath) | ConvertFrom-Json -ErrorAction Stop
    if ((Field $approval 'passed') -ne $true -or
        (Field $approval 'decision') -cne 'one_bounded_direct_HTTP11_immutable_private_snapshot_attempt' -or
        (Field $approval 'source_sha256') -cne $receipt.source_sha256 -or
        (Field $approval 'target_commit') -cne $target) {
        Fail-Safe 'unreviewed_source_or_target_commit'
    }
    $receipt.root_review_sha256 = (Get-FileHash -LiteralPath $reviewPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not (Test-Path -LiteralPath (Join-Path $snapshot '.git'))) { Fail-Safe 'snapshot_missing' }
    $receipt.local_head_before = Require-Git (Git-Call 'local_HEAD_before' @('rev-parse', 'HEAD')) 'local_HEAD_unavailable'
    if ($receipt.local_head_before -cne $target) { Fail-Safe 'immutable_HEAD_mismatch' }
    $branch = Require-Git (Git-Call 'local_branch' @('symbolic-ref', '--short', 'HEAD')) 'branch_unavailable'
    if ($branch -cne 'main') { Fail-Safe 'snapshot_branch_mismatch' }
    $url = Require-Git (Git-Call 'remote_url_metadata' @('remote', 'get-url', 'origin')) 'remote_url_unavailable'
    if ($url -cne $remoteUrl) { $url = $null; Fail-Safe 'remote_url_mismatch' }; $url = $null

    $credentials = Git-Call 'GCM_credential_fill' @('credential', 'fill') 30 ("protocol=https`nhost=github.com`nusername=$owner`n`n")
    $credentialOutput = Require-Git $credentials 'credential_unavailable'
    $credentials.output = $null
    $pairs = @{}
    foreach ($line in ($credentialOutput -split "`n")) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $Matches[1]; if ($pairs.ContainsKey($name)) { Fail-Safe 'credential_duplicate_fields' }
            $pairs[$name] = $Matches[2].TrimEnd("`r")
        }
    }
    if ($pairs['username'] -cne $owner -or $pairs['protocol'] -cne 'https' -or $pairs['host'] -cne 'github.com' -or
        [string]::IsNullOrWhiteSpace($pairs['password'])) { Fail-Safe 'credential_identity_mismatch' }
    $password = $pairs['password']; $pairs.Clear(); $credentialOutput = $null; $line = $null; $Matches = $null
    $script:apiHeaders = @{ Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2026-03-10'
        'User-Agent' = 'GeoBC-reviewed-immutable-publication'; Authorization = 'Bearer ' + $password }
    $password = $null
    $handler = [Net.Http.HttpClientHandler]::new(); $handler.UseProxy = $false
    $script:httpClient = [Net.Http.HttpClient]::new($handler, $true)
    $script:httpClient.Timeout = [TimeSpan]::FromSeconds(45)
    $receipt.verification_before = Verify-Identity 'before'
    $receipt.remote_main_before = Read-Remote-Main 'before'
    $receipt.route.route_actual_validation_pending = $false

    if ($receipt.remote_main_before -ceq $target) {
        $receipt.ancestor_check = 'already_exact_target'; $receipt.remote_outcome = 'target_already_observed'
    } else {
        if ($null -ne $receipt.remote_main_before) {
            $object = Git-Call 'remote_commit_known_locally' @('cat-file', '-e', ($receipt.remote_main_before + '^{commit}'))
            if ($object.timeout -or $object.exit_code -ne 0) { Fail-Safe 'remote_commit_unknown_locally_no_fetch' }
            $ancestor = Git-Call 'fast_forward_ancestor_check' @('merge-base', '--is-ancestor', $receipt.remote_main_before, $target)
            if ($ancestor.timeout -or $ancestor.exit_code -ne 0) { Fail-Safe 'remote_not_verified_ancestor' }
            $receipt.ancestor_check = 'known_local_ancestor_exit0'
        } else { $receipt.ancestor_check = 'verified_empty_main_ref' }
        # Explicit SHA refspec, ordinary fast-forward only. No branch/config update.
        $receipt.push_attempted = $true; $receipt.remote_outcome = 'unknown_after_mutation_attempt'
        $pushResult = Git-Call 'ordinary_push_immutable_SHA' @('push', '--porcelain', 'origin', ($target + ':refs/heads/main')) 300
        $receipt.push_exit_code = $pushResult.exit_code
        $pushResult.output = $null
    }

    # Postchecks are independent: REST still runs when the remote-ref check fails.
    $postFailures = [Collections.Generic.List[string]]::new()
    try { $receipt.remote_main_after = Read-Remote-Main 'after' } catch {
        $postFailures.Add($script:safeFailure); $script:safeFailure = $null
    }
    try { $receipt.verification_after = Verify-Identity 'after' } catch {
        $postFailures.Add($script:safeFailure); $script:safeFailure = $null
    }
    $receipt.local_head_after = Require-Git (Git-Call 'local_HEAD_after' @('rev-parse', 'HEAD')) 'local_HEAD_unavailable_after'
    if ($receipt.local_head_after -cne $target) { Fail-Safe 'immutable_HEAD_changed' }
    if ($postFailures.Count -gt 0) { Fail-Safe ('post_verification_incomplete:' + ($postFailures -join ',')) }
    if ($receipt.remote_main_after -cne $target) { Fail-Safe 'remote_target_not_confirmed' }
    $receipt.remote_outcome = 'exact_target_and_private_identity_confirmed'
    $receipt.completed = $true; $receipt.status = 'completed'
} catch {
    $receipt.status = 'failed'
    $receipt.failure_category = if ($null -ne $script:safeFailure) { $script:safeFailure } else { 'local_runtime_failure' }
    if ($receipt.push_attempted -and -not $receipt.completed) { $receipt.remote_outcome = 'unknown_or_not_confirmed_after_mutation_attempt' }
} finally {
    $credentialOutput = $null; $password = $null
    if ($null -ne $script:apiHeaders) { $script:apiHeaders.Clear(); $script:apiHeaders = $null }
    if ($null -ne $script:httpClient) { $script:httpClient.Dispose(); $script:httpClient = $null }
    $receipt.elapsed_seconds = $watch.Elapsed.TotalSeconds; $receipt.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    # Receipt outside the immutable snapshot. Never touch its tracked or untracked files.
    $temporary = $receiptPath + '.tmp'
    [IO.File]::WriteAllText($temporary, ($receipt | ConvertTo-Json -Depth 16) + "`n", [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
}

# Only sanitized receipt data reaches stdout; no raw process/HTTP payloads.
$receipt | ConvertTo-Json -Depth 16
if (-not $receipt.completed) { exit 1 }
exit 0
