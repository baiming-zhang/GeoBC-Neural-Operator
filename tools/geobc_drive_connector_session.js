// Run in Codex functions.exec, where tools, text, and store are supplied.
// This acquisition caller is separate from the frozen scientific 20-source map.
// File-reference URLs remain in memory and stdin; never print or persist them.
const workspace = "C:/Users/ASUS/Desktop/GeoBC Neural Operator";
const python = "C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const plan = "notes/geobc_coverage_comparison_plan.json";
const session = {
  kind: "fixed_DeepJEB_GoogleDrive_connector_binary_recovery",
  started_at_utc: new Date().toISOString(),
  concurrency: 2,
  completed_this_session: 0,
  failed_this_session: 0,
  consecutive_failed_files: 0,
  stop_scheduling: false,
  scientific_array_reads: 0,
  new_DEV_labels_sealed: true,
  active: [],
};
store("geobc_drive_recovery_session", session);

function unwrap(result) {
  if (result?.isError) throw new Error("connector_tool_error");
  if (result?.structuredContent) return result.structuredContent;
  for (const block of result?.content ?? []) {
    if (block.type === "text") {
      try { return JSON.parse(block.text); } catch (_) {}
    }
  }
  throw new Error("connector_metadata_shape_unavailable");
}

async function cli(action, payload = null, materialize = false) {
  const module = materialize
    ? "tools.materialize_geobc_coverage_drive_ref"
    : "tools.geobc_drive_connector_cli";
  // PowerShell's literal here-string performs no expansion of the JSON.
  const input = payload === null ? "" : `@'\n${JSON.stringify(payload)}\n'@ | `;
  const cmd = `${input}& '${python}' -u -m ${module} --plan '${plan}'${materialize ? "" : ` --action ${action}`}`;
  let result = await tools.exec_command({cmd, workdir: workspace, max_output_tokens: 5000});
  let output = result.output ?? "";
  while (result.session_id) {
    result = await tools.write_stdin({session_id: result.session_id, chars: "", yield_time_ms: 10000, max_output_tokens: 5000});
    output += result.output ?? "";
  }
  // Never echo raw command errors: they can contain a temporary file URL.
  if (result.exit_code !== 0) throw new Error(materialize ? "local_materialization_failed" : `recovery_${action}_failed`);
  try { return JSON.parse(output.trim()); }
  catch (_) { throw new Error("recovery_stdout_not_JSON"); }
}

const initial = await cli("pending");
const pending = [];
if (!Number.isSafeInteger(initial.pending_count)) throw new Error("fixed_pending_inventory_unavailable");
for (let offset = 0; offset < initial.pending_count; offset += 16) {
  const page = await cli("pending_page", {offset, limit: 16});
  if (!Array.isArray(page.pending)) throw new Error("fixed_pending_page_unavailable");
  pending.push(...page.pending);
}
if (pending.length !== initial.pending_count || pending.some(x => x.reserved_open))
  throw new Error("pending_snapshot_incomplete_or_reservation_unresolved");
session.pending_at_start = pending.length;
text({phase: "connector_binary_recovery_started", pending: pending.length, concurrency: 2, scientific_array_reads: 0});
let cursor = 0;

async function worker() {
  while (!session.stop_scheduling && cursor < pending.length) {
    const entry = pending[cursor++];
    const active = {relative_path: entry.relative_path, phase: "metadata", reservation_id: null};
    session.active.push(active);
    store("geobc_drive_recovery_session", session);
    let reservationId = null;
    let fetchStart = null;
    let fetchEnd = null;
    let fetchElapsed = null;
    try {
      const metadata = unwrap(await tools.mcp__codex_apps__google_drive_get_file_metadata({
        fileId: entry.drive_file_id,
        fields: "id,name,mimeType,size,modifiedTime",
      }));
      const size = Number(metadata.size);
      if (metadata.id !== entry.drive_file_id || (metadata.title ?? metadata.name) !== entry.name
          || !Number.isSafeInteger(size) || size <= 0) throw new Error("official_file_metadata_identity_mismatch");
      const sourceMetadata = {
        id: metadata.id, name: metadata.title ?? metadata.name,
        size, modified_time: metadata.modified_time ?? null,
      };
      const reserved = await cli("reserve", {relative_path: entry.relative_path, source_metadata: sourceMetadata});
      reservationId = reserved.reservation_id ?? reserved.attempt_id;
      if (typeof reservationId !== "string" || !reservationId) throw new Error("reservation_id_unavailable");
      active.reservation_id = reservationId;
      active.phase = "connector_fetch";
      store("geobc_drive_recovery_session", session);
      fetchStart = new Date().toISOString();
      const startMs = Date.now();
      const rawResult = await tools.mcp__codex_apps__google_drive_fetch({
        url: entry.source_url, download_raw_file: true, include_base64: false,
      });
      fetchElapsed = (Date.now() - startMs) / 1000;
      fetchEnd = new Date().toISOString();
      const raw = unwrap(rawResult);
      if (raw.id !== entry.drive_file_id || Number(raw.file_size_bytes) !== size
          || (raw.title ?? entry.name) !== entry.name || typeof raw.file_uri?.download_url !== "string")
        throw new Error("raw_file_reference_identity_mismatch");
      sourceMetadata.artifact_file_id = raw.file_uri.file_id ?? null;
      sourceMetadata.modified_time = raw.modified_time ?? sourceMetadata.modified_time;
      active.phase = "local_materialization";
      store("geobc_drive_recovery_session", session);
      await cli(null, {
        reservation_id: reservationId,
        relative_path: entry.relative_path,
        file_uri: raw.file_uri.download_url,
        source_metadata: sourceMetadata,
        fetch_started_at_utc: fetchStart, fetched_at_utc: fetchEnd,
        fetch_elapsed_seconds: fetchElapsed,
      }, true);
      session.completed_this_session++;
      session.consecutive_failed_files = 0;
    } catch (error) {
      session.failed_this_session++;
      session.consecutive_failed_files++;
      const allowedReasons = new Set([
        "connector_tool_error", "connector_metadata_shape_unavailable",
        "official_file_metadata_identity_mismatch", "reservation_id_unavailable",
        "raw_file_reference_identity_mismatch", "local_materialization_failed",
        "recovery_reserve_failed", "recovery_stdout_not_JSON",
      ]);
      const reason = allowedReasons.has(error?.message) ? error.message : "connector_recovery_operation_failed";
      if (reservationId) {
        try {
          await cli("fail", {reservation_id: reservationId, reason,
            fetch_started_at_utc: fetchStart, fetched_at_utc: fetchEnd,
            fetch_elapsed_seconds: fetchElapsed});
        } catch (_) { session.stop_scheduling = true; }
      }
      text({phase: "fixed_file_failure", relative_path: entry.relative_path, reason});
      // A shared route failure should be investigated before more requests.
      if (session.consecutive_failed_files >= 3) session.stop_scheduling = true;
    } finally {
      session.active = session.active.filter(x => x !== active);
      session.updated_at_utc = new Date().toISOString();
      store("geobc_drive_recovery_session", session);
      const finished = session.completed_this_session + session.failed_this_session;
      if (finished % 10 === 0) text({phase: "binary_recovery_progress",
        completed: session.completed_this_session, failed: session.failed_this_session,
        pending_at_start: session.pending_at_start, active: session.active,
        elapsed_seconds: (Date.now() - Date.parse(session.started_at_utc)) / 1000});
    }
  }
}

await Promise.all([worker(), worker()]);
const final = await cli("finalize");
session.completed_at_utc = new Date().toISOString();
session.final_status = final.status ?? null;
store("geobc_drive_recovery_session", session);
text({phase: "binary_recovery_session_finished", completed: session.completed_this_session,
  failed: session.failed_this_session, stopped_early: session.stop_scheduling,
  aggregate_status: session.final_status, scientific_array_reads: 0, new_DEV_labels_sealed: true});
