param(
    [switch]$Execute,
    [string]$ExpectedSourceSHA
)

# One bounded transport diagnostic only. No archive/scientific fields are parsed.
# Signed redirected URIs and exception messages remain in memory and are never logged.
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceSHA = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $Execute) {
    @{ status='source_ready_only'; source_sha256=$sourceSHA; network_executed=$false } |
        ConvertTo-Json -Compress
    exit 0
}
if ($ExpectedSourceSHA -notmatch '^[0-9a-fA-F]{64}$' -or
    $ExpectedSourceSHA.ToLowerInvariant() -ne $sourceSHA) {
    @{ status='failed'; error_code='explicit_reviewed_source_SHA_required'; network_executed=$false } |
        ConvertTo-Json -Compress
    exit 2
}

$deadline = [DateTimeOffset]::Parse('2026-10-10T08:15:02Z')
$byteLimit = 161061273600L
$priorHFReservations = 4194304L
$bodyCap = 2097152L
$expectedPrefixBytes = 1277224960L
$archiveBytes = 53817481481L
$revision = '160424ca2a5095315ff9ead8a5fc36d2667fd3c6'
$prefixPath = Join-Path $project "data/raw/.download_cache/160424ca2a50_FieldMesh.tar.gz.part"
$recoveryDir = Join-Path $project 'data/raw_geobc_coverage/.connector_recovery'
$bodyPath = Join-Path $recoveryDir 'hf_dotnet_FieldMesh_probe.bin'
$preflightPath = Join-Path $project 'notes/geobc_hf_dotnet_preflight.json'
$receiptPath = Join-Path $project 'notes/geobc_hf_dotnet_probe_receipt.json'
$timer = [Diagnostics.Stopwatch]::StartNew()
$script:FailureCode = $null
$result = [ordered]@{
    kind='bounded_official_original_DeepJEB_HF_DotNet_transport_probe'
    status='failed'; stage='local_metadata'; source_sha256=$sourceSHA; network_executed=$false
    scientific_digest='bcce7d422749ca26b72ecba57ddcbbb985dcd3643ac77cf87f5671310b1aa01b'
    component='FieldMesh'; pinned_revision=$revision
    HTTP_body_bytes_observed=0L; file_bytes_written_observed=0L
    HEAD_attempts=0; GET_attempts=0; total_wall_cap_seconds=90
    phase_wall_cap_seconds=30; connect_timeout_seconds=30
    proxy_route='explicit_http_127.0.0.1_7877_only_for_this_client'
    TLS_policy='Windows_DotNet_default_Schannel_certificate_validation'
    logical_requests_per_phase_maximum=1; redirects_per_request_maximum=3
    physical_socket_count_one_not_guaranteed=$true
    prior_HF_reservation_bytes=$priorHFReservations; this_conditional_body_reservation_bytes=$bodyCap
    network_bytes_exact=$false; exact_historical_cost_proven=$false
    source_prefix_body_reads=0; gzip_or_archive_parsing=0
    CSV_or_label_or_GT_or_weight_reads=0; GPU_calls=0
    original_clock_or_scientific_source_pins_modified=$false
    signed_URI_saved=$false; exception_Message_saved=$false; response_headers_saved=$false
    full_archive_or_127GB_delivery_verified=$false
}
$reserved = $false
$client = $null; $handler = $null; $headResponse = $null; $getResponse = $null
$headRequest = $null; $getRequest = $null; $headCTS = $null; $getCTS = $null
$responseStream = $null; $fileStream = $null

function Write-AtomicJson([string]$Target, $Value, [bool]$NoOverwrite=$false) {
    $parent = [IO.Path]::GetDirectoryName($Target)
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    $tmp = Join-Path $parent ([IO.Path]::GetFileName($Target) + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $text = ConvertTo-Json -InputObject $Value -Depth 12
        [IO.File]::WriteAllText($tmp, $text + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($tmp, $Target, (-not $NoOverwrite))
    } finally {
        if ([IO.File]::Exists($tmp)) { [IO.File]::Delete($tmp) }
    }
}

function Fail-Guard([string]$Code) {
    $script:FailureCode = $Code
    throw [InvalidOperationException]::new($Code)
}

function New-PhaseToken {
    $left = [Math]::Min(30.0, [Math]::Min(
        ($deadline - [DateTimeOffset]::UtcNow).TotalSeconds,
        90.0 - $timer.Elapsed.TotalSeconds))
    if ($left -le 0) { Fail-Guard 'absolute_or_total_wall_deadline' }
    return [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($left))
}

function Check-Wall {
    if ($timer.Elapsed.TotalSeconds -ge 90 -or [DateTimeOffset]::UtcNow -ge $deadline) {
        Fail-Guard 'absolute_or_total_wall_deadline'
    }
}

try {
    if (-not $IsWindows -or $PSVersionTable.PSEdition -ne 'Core' -or
        -not ('System.Net.Http.SocketsHttpHandler' -as [type])) {
        Fail-Guard 'Windows_Core_SocketsHttpHandler_required'
    }
    if ([IO.File]::Exists($preflightPath) -or [IO.File]::Exists($bodyPath) -or
        [IO.File]::Exists($receiptPath)) { Fail-Guard 'one_probe_already_reserved_or_attempted' }
    Check-Wall
    $plan = Get-Content -Raw -LiteralPath (Join-Path $project 'notes/geobc_coverage_comparison_plan.json') |
        ConvertFrom-Json
    if ($plan.scientific_digest -ne $result.scientific_digest) { Fail-Guard 'frozen_scientific_digest_differs' }
    $clock = Get-Content -Raw -LiteralPath (Join-Path $recoveryDir 'clock.json') | ConvertFrom-Json
    if ($null -eq $clock.conditional_byte_charge -or $clock.network_bytes_exact -ne $false) {
        Fail-Guard 'honest_conditional_recovery_clock_required'
    }
    $prefix = Get-Item -LiteralPath $prefixPath
    if ($prefix.PSIsContainer -or $prefix.Length -ne $expectedPrefixBytes) { Fail-Guard 'original_prefix_stat_mismatch' }
    $charged = [long]$clock.conditional_byte_charge + $priorHFReservations + $bodyCap
    if ($charged -gt $byteLimit) { Fail-Guard 'conditional_byte_cap' }
    $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($project))
    if ($drive.AvailableFreeSpace -lt $byteLimit) { Fail-Guard 'minimum_free_disk_guard' }
    $from = $expectedPrefixBytes
    $to = $from + $bodyCap - 1
    $preflight = [ordered]@{
        status='reserved'; kind=$result.kind; source_sha256=$sourceSHA
        reserved_at_utc=[DateTimeOffset]::UtcNow.ToString('o'); PID=$PID
        scientific_digest=$result.scientific_digest; route=$result.proxy_route
        original_recovery_conditional_charge_bytes=[long]$clock.conditional_byte_charge
        prior_HF_reservation_bytes=$priorHFReservations; this_body_reservation_bytes=$bodyCap
        total_conditional_charge_with_this_probe_bytes=$charged; maximum_conditional_bytes=$byteLimit
        cost_exact=$false; network_bytes_exact=$false
        original_unknown_transfer_costs_preserved=$true
        conditional_charge_is_not_measured_network_total=$true
        minimum_free_bytes=$byteLimit; available_free_bytes=$drive.AvailableFreeSpace
        absolute_deadline_utc=$deadline.ToString('o'); total_wall_cap_seconds=90
        prefix_path='data/raw/.download_cache/160424ca2a50_FieldMesh.tar.gz.part'
        prefix_bytes=$prefix.Length; prefix_mtime_utc=$prefix.LastWriteTimeUtc.ToString('o')
        prefix_body_read=$false; archive_bytes=$archiveBytes
        range_from=$from; range_to=$to; body_cap_bytes=$bodyCap
        body_output='data/raw_geobc_coverage/.connector_recovery/hf_dotnet_FieldMesh_probe.bin'
        TLS_default_validation_preserved=$true; global_proxy_or_ServicePoint_changes=$false
        signed_URI_saved=$false; gzip_or_scientific_field_parsing=$false
    }
    Write-AtomicJson $preflightPath $preflight -NoOverwrite $true
    $reserved = $true
    $result.started_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    $result.total_conditional_charge_with_probe_bytes = $charged
    $result.range_from = $from; $result.range_to = $to

    $handler = [Net.Http.SocketsHttpHandler]::new()
    $handler.Proxy = [Net.WebProxy]::new('http://127.0.0.1:7877')
    $handler.UseProxy = $true; $handler.UseCookies = $false
    $handler.ConnectTimeout = [TimeSpan]::FromSeconds(30)
    $handler.MaxConnectionsPerServer = 1
    $handler.MaxAutomaticRedirections = 3
    $handler.MaxResponseDrainSize = 0
    $handler.AutomaticDecompression = [Net.DecompressionMethods]::None
    # No custom SslOptions or certificate callback: platform validation remains active.
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(30)
    $url = [Uri]("https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/resolve/" + $revision + '/FieldMesh.tar.gz')

    $result.stage = 'HEAD'; $headCTS = New-PhaseToken
    $headRequest = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Head, $url)
    $headTimer = [Diagnostics.Stopwatch]::StartNew()
    $result.HEAD_attempts = 1
    $result.network_executed = $true
    $headResponse = $client.SendAsync($headRequest,
        [Net.Http.HttpCompletionOption]::ResponseHeadersRead,
        $headCTS.Token).GetAwaiter().GetResult()
    $result.HEAD_elapsed_seconds = $headTimer.Elapsed.TotalSeconds
    $result.HEAD_status_code = [int]$headResponse.StatusCode
    $result.HEAD_content_length = $headResponse.Content.Headers.ContentLength
    if ($result.HEAD_status_code -ne 200 -or $result.HEAD_content_length -ne $archiveBytes) {
        Fail-Guard 'HEAD_200_exact_archive_size_not_confirmed'
    }
    # Use the returned endpoint only in memory. Never save or print a redirected URI.
    $finalUri = $headResponse.RequestMessage.RequestUri
    $hostName = $finalUri.DnsSafeHost.ToLowerInvariant()
    if ($finalUri.Scheme -ne 'https' -or $finalUri.UserInfo -or
        -not ($hostName -eq 'huggingface.co' -or $hostName.EndsWith('.huggingface.co') -or
            $hostName.EndsWith('.hf.co'))) {
        Fail-Guard 'HEAD_final_endpoint_outside_official_HTTPS_hosts'
    }
    $headResponse.Dispose(); $headResponse = $null
    Check-Wall

    $result.stage = 'Range_GET'; $getCTS = New-PhaseToken
    $getRequest = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $finalUri)
    $getRequest.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new($from, $to)
    $getRequest.Headers.TryAddWithoutValidation('Accept-Encoding', 'identity') | Out-Null
    $getTimer = [Diagnostics.Stopwatch]::StartNew()
    $result.GET_attempts = 1
    $getResponse = $client.SendAsync($getRequest,
        [Net.Http.HttpCompletionOption]::ResponseHeadersRead,
        $getCTS.Token).GetAwaiter().GetResult()
    $result.GET_status_code = [int]$getResponse.StatusCode
    $range = $getResponse.Content.Headers.ContentRange
    if ($result.GET_status_code -ne 206 -or $null -eq $range -or $range.Unit -ne 'bytes' -or
        $range.From -ne $from -or $range.To -ne $to -or $range.Length -ne $archiveBytes -or
        $getResponse.Content.Headers.ContentLength -ne $bodyCap -or
        $getResponse.Content.Headers.ContentEncoding.Count -ne 0) {
        Fail-Guard 'exact_206_Range_and_length_not_confirmed'
    }
    $responseStream = $getResponse.Content.ReadAsStreamAsync($getCTS.Token).GetAwaiter().GetResult()
    $fileStream = [IO.FileStream]::new($bodyPath, [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read, 65536, $true)
    $buffer = [byte[]]::new(65536)
    while ($result.HTTP_body_bytes_observed -lt $bodyCap) {
        Check-Wall
        $count = [Math]::Min($buffer.Length, $bodyCap - $result.HTTP_body_bytes_observed)
        $n = $responseStream.ReadAsync($buffer, 0, [int]$count, $getCTS.Token).GetAwaiter().GetResult()
        if ($n -eq 0) { Fail-Guard 'Range_body_truncated' }
        $result.HTTP_body_bytes_observed += $n
        $fileStream.WriteAsync($buffer, 0, $n, $getCTS.Token).GetAwaiter().GetResult()
        $result.file_bytes_written_observed += $n
    }
    $fileStream.FlushAsync($getCTS.Token).GetAwaiter().GetResult()
    $result.GET_elapsed_seconds = $getTimer.Elapsed.TotalSeconds
    $result.GET_MB_per_second = ($result.HTTP_body_bytes_observed / 1000000.0) / $result.GET_elapsed_seconds
    Check-Wall
    $result.status = 'completed'; $result.stage = 'bounded_Range_complete'
} catch {
    $result.error_code = if ($script:FailureCode) { $script:FailureCode } else { 'DotNet_transport_or_IO_exception' }
    $chain = @(); $errorObject = $_.Exception
    for ($i=0; $errorObject -and $i -lt 6; $i++) {
        $item = [ordered]@{ type=$errorObject.GetType().FullName; HResult=[int]$errorObject.HResult }
        if ($errorObject -is [Net.Sockets.SocketException]) {
            $item.SocketErrorCode = [int]$errorObject.SocketErrorCode
            $item.NativeErrorCode = [int]$errorObject.NativeErrorCode
        } elseif ($errorObject -is [ComponentModel.Win32Exception]) {
            $item.NativeErrorCode = [int]$errorObject.NativeErrorCode
        }
        $chain += $item; $errorObject = $errorObject.InnerException
    }
    $result.exception_chain = $chain
} finally {
    foreach ($resource in @($fileStream,$responseStream,$getResponse,$headResponse,
            $getRequest,$headRequest,$getCTS,$headCTS,$client,$handler)) {
        if ($null -ne $resource) {
            try { $resource.Dispose() } catch { $result.resource_dispose_failure_type=$_.Exception.GetType().FullName }
        }
    }
    if ($null -ne $headTimer -and -not $result.Contains('HEAD_elapsed_seconds')) {
        $result.HEAD_elapsed_seconds = $headTimer.Elapsed.TotalSeconds
    }
    if ($null -ne $getTimer -and -not $result.Contains('GET_elapsed_seconds')) {
        $result.GET_elapsed_seconds = $getTimer.Elapsed.TotalSeconds
    }
    if ($result.Contains('GET_elapsed_seconds') -and $result.GET_elapsed_seconds -gt 0) {
        $result.GET_observed_MB_per_second = ($result.HTTP_body_bytes_observed / 1000000.0) / $result.GET_elapsed_seconds
    }
    $result.elapsed_seconds = $timer.Elapsed.TotalSeconds
    $result.completed_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    if ([IO.File]::Exists($bodyPath)) {
        $result.retained_file_bytes = [IO.FileInfo]::new($bodyPath).Length
        $result.failed_partial_retained = $result.status -ne 'completed'
    }
    if ($reserved) {
        try { Write-AtomicJson $receiptPath $result -NoOverwrite $true }
        catch {
            $result.status = 'failed'
            $result.receipt_write_failure_type = $_.Exception.GetType().FullName
        }
    }
    $url = $null; $finalUri = $null; $hostName = $null
}
$result | ConvertTo-Json -Depth 12 -Compress
if ($result.status -eq 'completed') { exit 0 } else { exit 2 }
