[CmdletBinding()]
param([switch]$PublishAPI,
      [ValidatePattern('^$|^[0-9a-f]{40}$')][string]$TargetCommit = '',
      [ValidatePattern('^$|^[0-9a-f]{40}$')][string]$BaseCommit = '',
      [ValidateRange(0,5000)][int]$ExpectedChanges = 0)

# The default branch performs no filesystem, Git, credential, or network I/O.
if (-not $PublishAPI) {
    [ordered]@{ mode = 'source-only'; mutation_attempted = $false
        execution_requires_root_source_and_exact_commit_review = $true
        route = 'GitHub Git Data REST; direct HTTP/1.1; not executed'
        exact_commit_required = $true; force_update = $false } | ConvertTo-Json
    exit 0
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion -lt [Version]'7.2') {
    throw 'Actual execution requires PowerShell 7.2+.'
}
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$snapshot = Join-Path $root 'publication/GeoBC-Neural-Operator'
$reviewPath = Join-Path $root 'notes/github_gitdata_publication_root_review.json'
$target = $TargetCommit
$base = $BaseCommit
$owner = 'baiming-zhang'
$repoName = 'GeoBC-Neural-Operator'
$userId = [long]174442095
$repoId = [long]1410470321
$expectedChangeCount = $ExpectedChanges
$apiPrefix = '/repos/' + $owner + '/' + $repoName
$remoteUrl = 'https://github.com/baiming-zhang/GeoBC-Neural-Operator.git'
$strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
$receiptPath = $null
$script:safeFailure = $null
$script:apiHeaders = $null
$script:httpClient = $null
$script:requestCount = 0
$script:receiptWritable = $false
$credentialOutput = $null
$password = $null
$watch = [Diagnostics.Stopwatch]::StartNew()
$receipt = [ordered]@{
    kind = 'reviewed_exact_immutable_private_snapshot_gitdata_publication'
    status = 'running'; completed = $false
    started_at_utc = [DateTime]::UtcNow.ToString('o'); finished_at_utc = $null
    source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    root_review_sha256 = $null; attempt_id = $null
    target_commit = $target; base_commit = $base; expected_changed_files = $expectedChangeCount
    expected_identity = @{ owner = $owner; user_id = $userId; repository_id = $repoId; private = $true }
    route = @{ direct = $true; HTTP_version = '1.1'; REST_proxy = 'UseProxy=false'
        redirects_allowed = $false; TLS_policy_changes = 0; global_proxy_changes = 0
        REST_hard_timeout_seconds = 45; local_Git_hard_timeout_seconds = 45
        GCM_hard_timeout_seconds = 30; automatic_retry_count = 0 }
    local_head_before = $null; local_head_after = $null
    local_target_tree = $null; local_base_tree = $null
    local_raw_commit_sha_verified = $false; representable_raw_commit_verified = $false
    local_changed_files = $null; local_deleted_files = 0
    local_blob_bytes = 0; locally_verified_blobs = 0
    verification_before = $null; verification_before_ref_update = $null; verification_after = $null
    remote_main_before = $null; remote_main_before_ref_update = $null; remote_main_after = $null
    created_blob_requests = 0; created_blob_requests_verified = 0
    created_blob_requests_unconfirmed = 0; UTF8_blob_requests = 0; base64_blob_requests = 0
    tree_creation_attempted = $false; created_tree_sha = $null; exact_tree_verified = $false
    commit_creation_attempted = $false; created_commit_sha = $null; exact_commit_verified = $false
    ref_update_attempted = $false; ref_update_response_confirmed = $false
    remote_outcome = 'not_observed'; object_outcome = 'none_attempted'
    operations = [Collections.Generic.List[object]]::new(); failure_category = $null
    immutable_local_checkout = $true; local_staging_operations = 0; local_commit_operations = 0
    Git_network_operations = 0; fetch_operations = 0; push_operations = 0
    repository_creations = 0; force_update = $false
    credentials_written_by_script = $false; raw_response_persisted = $false
    raw_process_output_persisted = $false; elapsed_seconds = $null
}

function Fail-Safe([string]$Category) {
    $script:safeFailure = $Category
    throw 'Stopped; only an allowlisted category is recorded.'
}

function Field($Object, [string]$Name, $Default = $null) {
    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Save-Receipt {
    if (-not $script:receiptWritable) { return }
    $receipt.elapsed_seconds = $watch.Elapsed.TotalSeconds
    $temporary = $receiptPath + '.tmp'
    [IO.File]::WriteAllText($temporary, ($receipt | ConvertTo-Json -Depth 12) + "`n", [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
}

function Classify-ProcessFailure([string]$Raw) {
    if ($Raw -match 'Authentication failed|could not read Username|invalid credentials') { return 'authentication' }
    if ($Raw -match 'not a git repository|bad object|invalid object|unknown revision') { return 'local_Git_object' }
    if ($Raw -match 'permission.*denied|not authorized') { return 'permission' }
    return 'unspecified'
}

# Binary stdout avoids newline/encoding conversions in raw commits and blobs.
# Every invocation is local Git metadata/object reading, except explicit GCM fill.
function Git-Bytes([string]$Operation, [string[]]$Arguments,
                   [int]$TimeoutSeconds = 45, [string]$InputText = $null) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $gitExecutable
    foreach ($arg in (@('-C', $snapshot) + $Arguments)) { [void]$info.ArgumentList.Add($arg) }
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true; $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $info.StandardErrorEncoding = [Text.UTF8Encoding]::new($false)
    foreach ($name in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy')) {
        [void]$info.Environment.Remove($name)
    }
    foreach ($name in @('GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE','GCM_TRACE','GCM_TRACE_SECRETS')) {
        $info.Environment[$name] = '0'
    }
    $info.Environment['GIT_TERMINAL_PROMPT'] = '0'; $info.Environment['GCM_INTERACTIVE'] = 'never'
    $info.Environment['GIT_OPTIONAL_LOCKS'] = '0'
    $info.Environment['GIT_NO_REPLACE_OBJECTS'] = '1'
    $info.Environment['GIT_NO_LAZY_FETCH'] = '1'
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    $stream = [IO.MemoryStream]::new()
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false; $exitCode = $null; $rawErr = $null; $category = 'unspecified'
    try {
        [void]$process.Start()
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stream)
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
        if ($errTask.IsCompletedSuccessfully) { $rawErr = $errTask.Result }
        $category = Classify-ProcessFailure $rawErr
        if ($timedOut) { $category = 'hard_timeout' }
        $receipt.operations.Add(@{ operation = $Operation; exit_code = $exitCode; timeout = $timedOut
            elapsed_seconds = $timer.Elapsed.TotalSeconds; category = $category })
        if ($timedOut -or $exitCode -ne 0 -or -not $outTask.IsCompletedSuccessfully) {
            Fail-Safe ('local_operation_failed:' + $Operation + ':' + $category)
        }
        return ,($stream.ToArray())
    } finally {
        $rawErr = $null; $InputText = $null; $stream.Dispose(); $process.Dispose()
    }
}

function Git-Text([string]$Operation, [string[]]$Arguments) {
    $bytes = Git-Bytes $Operation $Arguments
    try { return $strictUtf8.GetString($bytes).TrimEnd("`r", "`n") }
    catch { Fail-Safe 'local_metadata_not_UTF8' }
    finally { $bytes = $null }
}

function Git-Object-SHA([string]$Kind, [byte[]]$Bytes) {
    $header = [Text.Encoding]::ASCII.GetBytes($Kind + ' ' + $Bytes.Length.ToString([Globalization.CultureInfo]::InvariantCulture) + [char]0)
    $hash = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA1)
    try {
        $hash.AppendData($header); $hash.AppendData($Bytes)
        return [Convert]::ToHexString($hash.GetHashAndReset()).ToLowerInvariant()
    } finally { $hash.Dispose() }
}

function Parse-Ident([string]$Raw) {
    $match = [regex]::Match($Raw, '\A(?<name>[^<>\r\n]+) <(?<email>[^<>\r\n]+)> (?<epoch>0|[1-9][0-9]*) (?<offset>[+-][0-9]{4})\z')
    if (-not $match.Success) { Fail-Safe 'raw_commit_ident_not_representable' }
    $offset = $match.Groups['offset'].Value
    $hours = [int]$offset.Substring(1, 2); $minutes = [int]$offset.Substring(3, 2)
    if ($hours -gt 14 -or $minutes -gt 59 -or ($hours -eq 14 -and $minutes -ne 0) -or $offset -ceq '-0000') {
        Fail-Safe 'raw_commit_offset_not_representable'
    }
    $totalMinutes = $hours * 60 + $minutes
    if ($offset[0] -eq '-') { $totalMinutes = -$totalMinutes }
    $epoch = [long]$match.Groups['epoch'].Value
    $date = [DateTimeOffset]::FromUnixTimeSeconds($epoch).ToOffset([TimeSpan]::FromMinutes($totalMinutes))
    $iso = $date.ToString("yyyy-MM-dd'T'HH:mm:sszzz", [Globalization.CultureInfo]::InvariantCulture)
    # Verify that API-representable fields reconstruct this original ident byte-for-byte.
    $reconstructed = $match.Groups['name'].Value + ' <' + $match.Groups['email'].Value + '> ' +
        $date.ToUnixTimeSeconds().ToString([Globalization.CultureInfo]::InvariantCulture) + ' ' + $date.ToString('zzz', [Globalization.CultureInfo]::InvariantCulture).Replace(':', '')
    if ($reconstructed -cne $Raw) { Fail-Safe 'raw_commit_ident_roundtrip_mismatch' }
    return @{ name = $match.Groups['name'].Value; email = $match.Groups['email'].Value; date = $iso }
}

function Parse-Raw-Commit([byte[]]$Bytes) {
    if ((Git-Object-SHA 'commit' $Bytes) -cne $target) { Fail-Safe 'local_raw_commit_SHA_mismatch' }
    $receipt.local_raw_commit_sha_verified = $true
    try { $raw = $strictUtf8.GetString($Bytes) } catch { Fail-Safe 'raw_commit_not_UTF8' }
    # Extra headers, multiple parents, signatures, and alternate encodings cannot be silently discarded.
    $match = [regex]::Match($raw, '(?s)\Atree (?<tree>[0-9a-f]{40})\nparent (?<parent>[0-9a-f]{40})\nauthor (?<author>[^\n]+)\ncommitter (?<committer>[^\n]+)\n\n(?<message>.*)\z')
    if (-not $match.Success -or $match.Groups['parent'].Value -cne $base -or
        $raw.Contains([char]0) -or -not $match.Groups['message'].Value.EndsWith("`n")) {
        Fail-Safe 'raw_commit_headers_or_message_not_representable'
    }
    $author = Parse-Ident $match.Groups['author'].Value
    $committer = Parse-Ident $match.Groups['committer'].Value
    $roundtrip = 'tree ' + $match.Groups['tree'].Value + "`nparent " + $base + "`nauthor " +
        $match.Groups['author'].Value + "`ncommitter " + $match.Groups['committer'].Value + "`n`n" + $match.Groups['message'].Value
    if ((Git-Object-SHA 'commit' $strictUtf8.GetBytes($roundtrip)) -cne $target) {
        Fail-Safe 'raw_commit_reconstruction_SHA_mismatch'
    }
    $receipt.representable_raw_commit_verified = $true
    return @{ tree = $match.Groups['tree'].Value; parents = @($base)
        author = $author; committer = $committer; message = $match.Groups['message'].Value }
}

function Parse-Changes([byte[]]$Bytes) {
    try { $raw = $strictUtf8.GetString($Bytes) } catch { Fail-Safe 'changed_paths_not_UTF8' }
    if (-not $raw.EndsWith([char]0)) { Fail-Safe 'local_raw_diff_shape' }
    $fields = $raw.Split([char]0)
    if (($fields.Length - 1) % 2 -ne 0) { Fail-Safe 'local_raw_diff_shape' }
    $entries = [Collections.Generic.List[object]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    for ($i = 0; $i -lt $fields.Length - 1; $i += 2) {
        $match = [regex]::Match($fields[$i], '\A:(?<oldmode>[0-9]{6}) (?<mode>[0-9]{6}) (?<oldsha>[0-9a-f]{40}) (?<sha>[0-9a-f]{40}) (?<status>[AMDT])\z')
        if (-not $match.Success) { Fail-Safe 'local_raw_diff_entry_shape' }
        $path = $fields[$i + 1]
        if ([string]::IsNullOrWhiteSpace($path) -or -not $seen.Add($path) -or $path.StartsWith('/') -or
            $path.Contains('\') -or $path -match '[\x00-\x1f\x7f]' -or
            @($path.Split('/') | Where-Object { $_ -in @('', '.', '..', '.git') }).Count -gt 0) {
            Fail-Safe 'changed_path_rejected'
        }
        $deleted = $match.Groups['status'].Value -ceq 'D'
        $mode = if ($deleted) { $match.Groups['oldmode'].Value } else { $match.Groups['mode'].Value }
        if ($mode -notin @('100644', '100755', '120000')) { Fail-Safe 'non_blob_tree_change_rejected' }
        # Only the already reviewed code/scalar/figure snapshot is eligible; no raw data or checkpoints.
        if ($path -match '(?i)\A(?:data|runs|cache|\.cache)/' -or
            $path -match '(?i)(?:\A|/)\.env(?:\.|\z)' -or
            $path -match '(?i)\.(?:pt|pth|npy|npz|h5|hdf5|pkl|pickle|parquet|zip|7z|tar|gz)\z') {
            Fail-Safe 'raw_or_secret_blob_path_rejected'
        }
        $sha = if ($deleted) { $null } else { $match.Groups['sha'].Value }
        if (-not $deleted -and $sha -ceq ('0' * 40)) { Fail-Safe 'local_diff_blob_SHA_missing' }
        $entries.Add(@{ path = $path; mode = $mode; type = 'blob'; sha = $sha })
        if ($deleted) { $receipt.local_deleted_files++ }
    }
    if ($entries.Count -ne $expectedChangeCount) { Fail-Safe 'reviewed_change_count_mismatch' }
    $receipt.local_changed_files = $entries.Count
    return ,$entries
}

# There is no retry policy. A failed POST/PATCH is an unconfirmed remote mutation.
function Api-Call([string]$Method, [string]$Endpoint, [string]$Operation, $Payload = $null,
                  [int]$ExpectedCode = 200) {
    $script:requestCount++
    if ($script:requestCount -gt 90) { Fail-Safe 'fixed_API_request_limit_exceeded' }
    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Method), 'https://api.github.com' + $Endpoint)
    $request.Version = [Version]'1.1'; $request.VersionPolicy = [Net.Http.HttpVersionPolicy]::RequestVersionExact
    foreach ($name in $script:apiHeaders.Keys) { [void]$request.Headers.TryAddWithoutValidation($name, $script:apiHeaders[$name]) }
    $json = $null
    if ($null -ne $Payload) {
        $json = ConvertTo-Json -InputObject $Payload -Depth 12 -Compress
        $request.Content = [Net.Http.StringContent]::new($json, [Text.UTF8Encoding]::new($false), 'application/json')
    }
    $cancel = [Threading.CancellationTokenSource]::new(45000)
    $timer = [Diagnostics.Stopwatch]::StartNew(); $code = 0; $response = $null; $body = $null
    try {
        $response = $script:httpClient.SendAsync($request, $cancel.Token).GetAwaiter().GetResult()
        $code = [int]$response.StatusCode
        if ($code -ne $ExpectedCode) { Fail-Safe ('REST_HTTP_' + $code) }
        $body = $response.Content.ReadAsStringAsync($cancel.Token).GetAwaiter().GetResult()
        return ($body | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        if ($null -eq $script:safeFailure) { $script:safeFailure = 'REST_transport_or_response' }
        throw 'REST failed; raw response, token, and diagnostics withheld.'
    } finally {
        $receipt.operations.Add(@{ operation = $Operation; method = $Method; HTTP_code = $code
            elapsed_seconds = $timer.Elapsed.TotalSeconds; raw_response_persisted = $false })
        $json = $null; $body = $null
        if ($null -ne $response) { $response.Dispose() }
        $request.Dispose(); $cancel.Dispose(); Save-Receipt
    }
}

function Verify-Identity([string]$Phase) {
    $user = Api-Call 'GET' '/user' ('user_' + $Phase)
    $repo = Api-Call 'GET' $apiPrefix ('repository_' + $Phase)
    $repoOwner = Field $repo 'owner'
    if ((Field $user 'login') -cne $owner -or (Field $user 'id') -ne $userId -or
        (Field $repo 'id') -ne $repoId -or (Field $repo 'private') -isnot [bool] -or -not $repo.private -or
        (Field $repoOwner 'login') -cne $owner -or (Field $repoOwner 'id') -ne $userId -or
        (Field $repo 'name') -cne $repoName -or (Field $repo 'full_name') -cne ($owner + '/' + $repoName)) {
        Fail-Safe 'identity_or_privacy_mismatch'
    }
    return @{ passed = $true; owner = $owner; user_id = $userId; repository_id = $repoId; private = $true }
}

function Read-Main([string]$Phase) {
    $value = Api-Call 'GET' ($apiPrefix + '/git/ref/heads/main') ('main_' + $Phase)
    $object = Field $value 'object'
    $sha = Field $object 'sha'
    if ((Field $value 'ref') -cne 'refs/heads/main' -or (Field $object 'type') -cne 'commit' -or
        $sha -isnot [string] -or $sha -cnotmatch '\A[0-9a-f]{40}\z') { Fail-Safe 'remote_main_shape_mismatch' }
    return $sha
}

try {
    if ($target -cnotmatch '\A[0-9a-f]{40}\z' -or $base -cnotmatch '\A[0-9a-f]{40}\z' -or
        $target -ceq $base -or $expectedChangeCount -lt 1) { Fail-Safe 'exact_target_base_delta_required' }
    if (-not (Test-Path -LiteralPath $reviewPath -PathType Leaf)) { Fail-Safe 'root_review_missing' }
    $approval = [IO.File]::ReadAllText($reviewPath) | ConvertFrom-Json -ErrorAction Stop
    $identity = Field $approval 'required_identity'
    $attempt = Field $approval 'attempt_id'
    if ((Field $approval 'passed') -ne $true -or (Field $approval 'whole_source_reviewed') -ne $true -or
        (Field $approval 'decision') -cne 'one_bounded_exact_immutable_gitdata_API_publication_attempt' -or
        (Field $approval 'source_sha256') -cne $receipt.source_sha256 -or
        (Field $approval 'target_commit') -cne $target -or (Field $approval 'base_commit') -cne $base -or
        (Field $approval 'changed_file_count') -ne $expectedChangeCount -or
        (Field $identity 'owner') -cne $owner -or (Field $identity 'user_id') -ne $userId -or
        (Field $identity 'repository_id') -ne $repoId -or (Field $identity 'private') -ne $true -or
        $attempt -isnot [string] -or $attempt -cnotmatch '\A[a-z0-9][a-z0-9_-]{0,63}\z') {
        Fail-Safe 'unreviewed_source_identity_or_immutable_target'
    }
    $receipt.root_review_sha256 = (Get-FileHash -LiteralPath $reviewPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.attempt_id = $attempt
    $receiptPath = Join-Path $root ('notes/github_gitdata_publication_' + $target + '_' + $attempt + '_receipt.json')
    if ((Test-Path -LiteralPath $receiptPath) -or (Test-Path -LiteralPath ($receiptPath + '.tmp'))) {
        Fail-Safe 'prior_attempt_receipt_exists_no_automatic_replay'
    }
    # Claim a unique receipt before credentials or API calls. Other receipts are never overwritten.
    $claim = [IO.File]::Open($receiptPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $claim.Dispose(); $script:receiptWritable = $true; Save-Receipt
    if (-not (Test-Path -LiteralPath (Join-Path $snapshot '.git'))) { Fail-Safe 'snapshot_missing' }
    $gitExecutable = (Get-Command git -ErrorAction Stop).Source
    $receipt.local_head_before = Git-Text 'local_HEAD_before' @('rev-parse', '--verify', 'HEAD')
    if ($receipt.local_head_before -cne $target) { Fail-Safe 'immutable_HEAD_mismatch' }
    if ((Git-Text 'local_branch' @('symbolic-ref', '--short', 'HEAD')) -cne 'main') { Fail-Safe 'snapshot_branch_mismatch' }
    $url = Git-Text 'local_remote_metadata' @('remote', 'get-url', 'origin')
    if ($url -cne $remoteUrl) { $url = $null; Fail-Safe 'snapshot_remote_metadata_mismatch' }; $url = $null
    $rawCommit = Git-Bytes 'local_raw_commit' @('cat-file', 'commit', $target)
    $commitPayload = Parse-Raw-Commit $rawCommit; $rawCommit = $null
    $receipt.local_target_tree = Git-Text 'local_target_tree' @('rev-parse', '--verify', ($target + '^{tree}'))
    $receipt.local_base_tree = Git-Text 'local_base_tree' @('rev-parse', '--verify', ($base + '^{tree}'))
    if ($commitPayload.tree -cne $receipt.local_target_tree -or $receipt.local_base_tree -cnotmatch '\A[0-9a-f]{40}\z') {
        Fail-Safe 'local_commit_tree_mismatch'
    }
    $diff = Git-Bytes 'local_reviewed_delta' @('diff-tree', '--no-commit-id', '--raw', '--no-renames', '--no-abbrev', '-r', '-z', $base, $target)
    $entries = Parse-Changes $diff; $diff = $null
    # Verify every immutable local blob, and its bounded size, before retrieving credentials.
    foreach ($entry in $entries) {
        if ($null -eq $entry.sha) { continue }
        $size = Git-Text 'local_blob_size' @('cat-file', '-s', $entry.sha)
        if ($size -cnotmatch '\A(?:0|[1-9][0-9]*)\z' -or [long]$size -gt 20MB) { Fail-Safe 'reviewed_blob_size_rejected' }
        $bytes = Git-Bytes 'local_blob_preflight' @('cat-file', 'blob', $entry.sha)
        if ($bytes.Length -ne [long]$size -or (Git-Object-SHA 'blob' $bytes) -cne $entry.sha) { Fail-Safe 'local_blob_SHA_mismatch' }
        $receipt.locally_verified_blobs++; $receipt.local_blob_bytes += $bytes.Length; $bytes = $null
    }
    Save-Receipt

    $credentials = Git-Bytes 'GCM_credential_fill' @('credential', 'fill') 30 ("protocol=https`nhost=github.com`nusername=$owner`n`n")
    try { $credentialOutput = $strictUtf8.GetString($credentials) } finally { $credentials = $null }
    $pairs = @{}
    foreach ($line in ($credentialOutput -split "`n")) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $Matches[1]
            if ($pairs.ContainsKey($name)) { Fail-Safe 'credential_duplicate_fields' }
            $pairs[$name] = $Matches[2].TrimEnd("`r")
        }
    }
    if ($pairs['username'] -cne $owner -or $pairs['protocol'] -cne 'https' -or $pairs['host'] -cne 'github.com' -or
        [string]::IsNullOrWhiteSpace($pairs['password'])) { Fail-Safe 'credential_identity_mismatch' }
    $password = $pairs['password']; $pairs.Clear(); $credentialOutput = $null; $line = $null; $Matches = $null
    $script:apiHeaders = @{ Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2026-03-10'
        'User-Agent' = 'GeoBC-reviewed-exact-gitdata-publication'; Authorization = 'Bearer ' + $password }
    $password = $null
    $handler = [Net.Http.HttpClientHandler]::new(); $handler.UseProxy = $false; $handler.AllowAutoRedirect = $false
    # Keep .NET's default TLS certificate validation. No certificate callback or TLS bypass.
    $script:httpClient = [Net.Http.HttpClient]::new($handler, $true)
    $script:httpClient.Timeout = [TimeSpan]::FromSeconds(45)
    $receipt.verification_before = Verify-Identity 'before'
    $receipt.remote_main_before = Read-Main 'before'
    if ($receipt.remote_main_before -cne $base -and $receipt.remote_main_before -cne $target) {
        Fail-Safe 'remote_main_changed_from_reviewed_base'
    }
    if ($receipt.remote_main_before -ceq $target) {
        $already = Api-Call 'GET' ($apiPrefix + '/git/commits/' + $target) 'already_target_commit'
        if ((Field $already 'sha') -cne $target -or (Field (Field $already 'tree') 'sha') -cne $receipt.local_target_tree) {
            Fail-Safe 'already_target_commit_tree_mismatch'
        }
        $receipt.exact_commit_verified = $true; $receipt.exact_tree_verified = $true
        $receipt.remote_outcome = 'exact_target_already_observed'
    } else {
        $baseRemote = Api-Call 'GET' ($apiPrefix + '/git/commits/' + $base) 'reviewed_base_commit'
        if ((Field $baseRemote 'sha') -cne $base -or
            (Field (Field $baseRemote 'tree') 'sha') -cne $receipt.local_base_tree) { Fail-Safe 'remote_base_tree_mismatch' }
        $index = 0
        foreach ($entry in $entries) {
            $index++
            if ($null -eq $entry.sha) { continue }
            $bytes = Git-Bytes 'local_blob_for_API' @('cat-file', 'blob', $entry.sha)
            if ((Git-Object-SHA 'blob' $bytes) -cne $entry.sha -or $bytes.Length -gt 20MB) { Fail-Safe 'immutable_blob_changed_or_oversized' }
            $encoding = 'utf-8'; $content = $null
            try { $content = $strictUtf8.GetString($bytes) } catch { $encoding = 'base64'; $content = [Convert]::ToBase64String($bytes) }
            $bytes = $null
            $receipt.created_blob_requests++; $receipt.created_blob_requests_unconfirmed++
            $receipt.object_outcome = 'objects_creation_attempted_not_branch_publication'
            if ($encoding -ceq 'utf-8') { $receipt.UTF8_blob_requests++ } else { $receipt.base64_blob_requests++ }
            Save-Receipt
            $blob = Api-Call 'POST' ($apiPrefix + '/git/blobs') ('blob_' + $index) @{ content = $content; encoding = $encoding } 201
            $content = $null
            if ((Field $blob 'sha') -cne $entry.sha) { Fail-Safe 'created_blob_SHA_mismatch' }
            $receipt.created_blob_requests_verified++; $receipt.created_blob_requests_unconfirmed--; Save-Receipt
        }
        $receipt.tree_creation_attempted = $true; Save-Receipt
        $tree = Api-Call 'POST' ($apiPrefix + '/git/trees') 'tree_create' @{ base_tree = $receipt.local_base_tree; tree = @($entries.ToArray()) } 201
        $treeSHA = Field $tree 'sha'
        if ($treeSHA -is [string] -and $treeSHA -cmatch '\A[0-9a-f]{40}\z') { $receipt.created_tree_sha = $treeSHA }
        if ($receipt.created_tree_sha -cne $receipt.local_target_tree) { Fail-Safe 'created_tree_SHA_mismatch_ref_unchanged' }
        $receipt.exact_tree_verified = $true
        $receipt.commit_creation_attempted = $true; Save-Receipt
        $created = Api-Call 'POST' ($apiPrefix + '/git/commits') 'commit_create' $commitPayload 201
        $createdSHA = Field $created 'sha'
        if ($createdSHA -is [string] -and $createdSHA -cmatch '\A[0-9a-f]{40}\z') { $receipt.created_commit_sha = $createdSHA }
        # GitHub may normalize unsupported commit representations. NEVER publish a substituted SHA.
        if ($receipt.created_commit_sha -cne $target -or
            (Field (Field $created 'tree') 'sha') -cne $receipt.local_target_tree) { Fail-Safe 'created_commit_SHA_mismatch_ref_unchanged' }
        $receipt.exact_commit_verified = $true; Save-Receipt
        $receipt.verification_before_ref_update = Verify-Identity 'before_ref_update'
        $receipt.remote_main_before_ref_update = Read-Main 'before_ref_update'
        if ($receipt.remote_main_before_ref_update -cne $base -and $receipt.remote_main_before_ref_update -cne $target) {
            Fail-Safe 'remote_main_changed_before_ref_update'
        }
        if ((Git-Text 'local_HEAD_before_ref_update' @('rev-parse', '--verify', 'HEAD')) -cne $target) { Fail-Safe 'immutable_HEAD_changed_before_ref_update' }
        if ($receipt.remote_main_before_ref_update -ceq $base) {
            $receipt.ref_update_attempted = $true; $receipt.remote_outcome = 'unknown_after_ref_update_attempt'; Save-Receipt
            $updated = Api-Call 'PATCH' ($apiPrefix + '/git/refs/heads/main') 'main_fast_forward_exact_target' @{ sha = $target; force = $false } 200
            if ((Field $updated 'ref') -cne 'refs/heads/main' -or
                (Field (Field $updated 'object') 'type') -cne 'commit' -or
                (Field (Field $updated 'object') 'sha') -cne $target) { Fail-Safe 'ref_update_response_not_exact_target' }
            $receipt.ref_update_response_confirmed = $true; Save-Receipt
        }
    }

    # Independent final checks; a failing one does not suppress the other observations.
    $postFailures = [Collections.Generic.List[string]]::new()
    try { $receipt.remote_main_after = Read-Main 'after' }
    catch { $postFailures.Add($(if ($null -ne $script:safeFailure) { $script:safeFailure } else { 'remote_main_postcheck' })); $script:safeFailure = $null }
    try { $receipt.verification_after = Verify-Identity 'after' }
    catch { $postFailures.Add($(if ($null -ne $script:safeFailure) { $script:safeFailure } else { 'identity_postcheck' })); $script:safeFailure = $null }
    try {
        $receipt.local_head_after = Git-Text 'local_HEAD_after' @('rev-parse', '--verify', 'HEAD')
        if ($receipt.local_head_after -cne $target) { Fail-Safe 'immutable_HEAD_changed' }
    } catch { $postFailures.Add($(if ($null -ne $script:safeFailure) { $script:safeFailure } else { 'local_HEAD_postcheck' })); $script:safeFailure = $null }
    if ($postFailures.Count -gt 0) { Fail-Safe ('post_verification_incomplete:' + ($postFailures -join ',')) }
    if ($receipt.remote_main_after -cne $target) { Fail-Safe 'remote_exact_target_not_confirmed' }
    $receipt.remote_outcome = 'exact_target_private_identity_and_immutable_local_HEAD_confirmed'
    $receipt.completed = $true; $receipt.status = 'completed'
} catch {
    $receipt.status = 'failed'
    $receipt.failure_category = if ($null -ne $script:safeFailure) { $script:safeFailure } else { 'local_runtime_failure' }
    if ($receipt.ref_update_attempted -and -not $receipt.completed) { $receipt.remote_outcome = 'unknown_or_unconfirmed_after_ref_update_attempt' }
    elseif ($receipt.remote_main_before -ceq $base) { $receipt.remote_outcome = 'no_ref_update_attempted_base_observed_before' }
} finally {
    $credentialOutput = $null; $password = $null; $content = $null; $commitPayload = $null; $pairs = $null; $Matches = $null
    if ($null -ne $script:apiHeaders) { $script:apiHeaders.Clear(); $script:apiHeaders = $null }
    if ($null -ne $script:httpClient) { $script:httpClient.Dispose(); $script:httpClient = $null }
    $receipt.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    try { Save-Receipt } catch { $receipt.status = 'failed'; $receipt.completed = $false; $receipt.failure_category = 'receipt_persistence_failed' }
}

# Allowlisted categories, hashes, counts and timings only. No raw subprocess or HTTP output.
$receipt | ConvertTo-Json -Depth 12
if (-not $receipt.completed) { exit 1 }
exit 0
