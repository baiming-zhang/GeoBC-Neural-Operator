"""Metadata-reuse two accepted coarse cases, execute only remaining three.

The immutable elastic_coarse body owns the same mathematical protocol.
This wrapper only adds strict provenance, five-case accounting and a 900s
campaign clock including the parent's already spent time exactly once.
No old core entrypoint/contract or label-loading entrypoint is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

SOURCE = "geobc/elastic_coarse_extension.py"
PLAN = "notes/geobc_elastic_coarse_extension_plan.json"
KIND = "elastic_p1_derived_coarse_extension"
CASES = ("101_428/ver", "101_56/ver", "101_78/ver", "102_249/ver", "102_38/ver")
REUSED = ("101_428/ver", "102_249/ver")
NEW = ("101_56/ver", "101_78/ver", "102_38/ver")
CORE_SOURCE = "geobc/elastic_coarse.py"
SOURCES = (SOURCE, CORE_SOURCE, "geobc/elastic_crosscad.py", "geobc/elastic_pilot.py",
           "geobc/elastic_pilot_data.py", "geobc/continuous_query_geometry.py")
ZERO_KEYS = ("GT_reads", "displacement_reads", "stress_reads", "old_final_reads", "DEV_dia_reads",
             "sealed_new_test_reads", "optimizer_steps", "scaler_fits", "learned_weight_reads", "setup_K_guards_repeated")
SCIENCE_KEYS = ("recipe", "comparison", "cost", "selection_gate", "read_policy", "campaign_budget")


def helpers():
    # All three modules have stdlib-only top levels. No scientific loader is invoked here.
    from . import elastic_coarse as core
    from . import elastic_crosscad as cross
    from . import elastic_pilot as base
    return core, cross, base


def host_path(value):
    return helpers()[1].host_path(value)


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(host_path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def closed(receipt):
    return all(receipt.get(key) == 0 for key in ZERO_KEYS)


def pinned_json(record):
    require(sha(record["path"]) == record["sha256"], "Pinned metadata changed: " + record["path"])
    return read_json(record["path"])


def plan_science(plan):
    return {key: plan[key] for key in SCIENCE_KEYS}


def request_contract(plan_path=PLAN):
    """Six-source/small-JSON metadata contract; no core scope-two contract call."""
    plan = read_json(plan_path)
    require(plan["kind"] == KIND and tuple(plan["case_keys"]) == CASES
            and tuple(plan["reused_case_keys"]) == REUSED and tuple(plan["new_case_keys"]) == NEW,
            "Frozen five inventory/two-reuse/three-new order differs")
    parent = pinned_json(plan["core_plan"])
    root = pinned_json(plan["core_root_review"])
    old = pinned_json(plan["core_summary"])
    require(parent["kind"] == "elastic_p1_derived_coarse_screen" and tuple(parent["case_keys"]) == REUSED
            and parent["scientific_rules_frozen"] is True and parent["actual_execution_released"] is True,
            "Accepted immutable two-case parent plan required")
    for key in ("recipe", "comparison", "cost", "read_policy"):
        require(plan[key] == parent[key], "No mathematical/read/cost protocol changes permitted: " + key)
    parent_science = {key: parent[key] for key in ("recipe", "comparison", "cost", "selection_gate", "read_policy")}
    require(root["passed"] is True and root["source_sha256"] == parent["source_sha256"][CORE_SOURCE]
            and root["science_recipe_sha256"] == digest(parent_science)
            and root["base_evidence_decision"] == "proceed_frozen_P1_derived_coarse",
            "Authentic accepted parent source/math release differs")
    require(old["status"] == "completed" and old["kind"] == parent["kind"]
            and old["config"]["stage"] == parent["kind"] and tuple(old["case_keys"]) == REUSED
            and tuple(old["config"]["contract"]["case_keys"]) == REUSED
            and old["config"]["contract"]["plan_science"] == parent_science
            and old["config"]["contract"]["source_sha256"] == parent["source_sha256"]
            and old["selection_gate"]["passed"] is True and closed(old["read_receipt"]),
            "Frozen parent actual two-case gate/source/config must pass")
    parent_seconds = old["costs"]["cumulative_stage_wall_seconds"]
    budget = plan["campaign_budget"]
    require(finite(parent_seconds) and parent_seconds > 0
            and parent_seconds == plan["core_summary"]["actual_parent_stage_wall_seconds"] == budget["parent_actual_seconds"]
            and budget["maximum_cumulative_wall_seconds"] == 900 and budget["new_case_inventory"] == 3
            and budget["actual_new_reference_reads_expected_if_complete"] == 3
            and budget["parent_reference_arrays_reopened"] is False and budget["parent_costs_or_configs_modified"] is False
            and budget["coarse_setup_seconds_per_new_case"] == 60 and budget["solve_seconds_per_new_case"] == 300
            and budget["attempted_iterations_per_new_case"] == 10000 and budget["actual_GPU_peak_bytes"] == 24*1024**3,
            "Frozen parent-once 900s campaign budget differs")
    gate = plan["selection_gate"]
    require(gate["inventory_denominator"] == gate["both_methods_true_converged_required"]
            == gate["allN_same_solution_gate_required"] == gate["minimum_CADs_with_at_least25percent_total_cold_reduction"] == 5
            and gate["minimum_total_cold_reduction_fraction"] == .25 and gate["complete_cost_components_required"] is True,
            "Frozen all-five same-equation/cold-gain gate differs")
    physical_plan = pinned_json(plan["physical_plan"])
    physical_summary = pinned_json(plan["physical_summary"])
    require(physical_plan["kind"] == physical_summary["kind"] == "elastic_crosscad_screen"
            and physical_summary["status"] == "completed" and tuple(physical_summary["case_keys"]) == CASES,
            "Accepted five-CAD physical parent required")
    pins = {name: sha(name) for name in SOURCES}
    require(set(plan["source_sha256"]) == set(SOURCES)
            and all(plan["source_sha256"][name] == value for name, value in pins.items())
            and {key: value for key, value in pins.items() if key != SOURCE} == parent["source_sha256"],
            "Exact immutable core-five plus wrapper-six source pins required")
    records, reused = {}, {}
    for key in CASES:
        record = plan["cases"][key]
        physical = record["physical_record"]
        require(physical["key"] == key and physical["cad"] == key.split("/")[0], "Fixed original physical case scope differs")
        manifest = pinned_json(record["input_manifest"])
        operator = pinned_json(record["operator_summary"])
        baseline = pinned_json(record["base_summary"])
        require(manifest["status"] == "completed" and manifest["feasibility_pass"] is True
                and manifest["contract"] == physical and manifest["file_identity"] == record["input_manifest"]["file_identity"]
                and manifest["cache_file"] == record["input_manifest"]["cache_file"], "Accepted target-free input cache differs")
        require(operator["status"] == "completed" and operator["feasibility_pass"] is True
                and operator["config"] == {"contract": physical, "stage": "elastic_crosscad_operator"}
                and operator["input_manifest_sha256"] == record["input_manifest"]["sha256"]
                and operator["algebra_guards"]["K_applications"] == 8 and operator["arrays"] == record["operator_summary"]["arrays"],
                "Accepted immutable fine P2/eight-guard operator differs")
        require(baseline["status"] == "completed" and baseline["config"] == {"contract": physical, "stage": "elastic_crosscad_pcg"}
                and baseline["input_compatible"] is True and baseline["operator_compatible"] is True
                and baseline["numerically_converged"] is True and baseline["numerical_state"] == "converged"
                and finite(baseline["true_relative_residual"]) and baseline["true_relative_residual"] <= 1e-8
                and baseline["prediction"] == record["base_summary"]["prediction"], "Accepted converged zero numerical reference differs")
        costs = baseline["costs"]
        zero = costs["cold_case_accounted_wall_seconds"]-costs["GT_read_and_metrics_seconds"]
        inherited = costs["input_preparation_seconds"]+costs["operator_wall_seconds"]
        require(finite(zero) and zero > 0 and finite(inherited)
                and zero == record["base_summary"]["zero_cold_without_GT_seconds"]
                and inherited == record["base_summary"]["inherited_input_operator_seconds"], "Frozen zero/fine setup costs differ")
        require(physical_summary["cases"][key]["config"]["contract"] == physical,
                "Frozen screen/full-node case identity differs")
        if key in REUSED:
            require(host_path(record["case_output"]) == host_path(plan["reused_cases"][key]["path"]).parent,
                    "Reused case path must retain original core directory, not wrapper output")
            reused_item = pinned_json(plan["reused_cases"][key])
            require(reused_item == old["cases"][key] and reused_item["config"]["stage"] == "elastic_p1_derived_coarse_case"
                    and reused_item["config"]["contract"]["physical_record"] == physical
                    and reused_item["config"]["contract"]["source_sha256"] == pins[CORE_SOURCE]
                    and reused_item["numerically_converged"] is True and reused_item["true_relative_residual"] <= 1e-8
                    and reused_item["same_solution_passed"] is True and reused_item["costs"]["components_complete"] is True
                    and reused_item["costs"]["cold_cost_reduction_fraction"] >= .25 and closed(reused_item["read_receipt"]),
                    "Original two actual summaries/provenance/gates must remain immutable")
            require(parent["cases"][key]["physical_record"] == physical,
                    "Reused fine equation differs from original core plan")
            reused[key] = {"source": plan["reused_cases"][key], "original_config": reused_item["config"]}
        else:
            require(host_path(record["case_output"]) == host_path(plan["output"])/key.replace("/", "_"),
                    "Frozen new-case output scope differs")
        records[key] = record
    science = {"kind": KIND, "case_keys": list(CASES), "reused_case_keys": list(REUSED), "new_case_keys": list(NEW),
        "plan_science": plan_science(plan), "records": records, "reused_sources": reused,
        "core_plan": plan["core_plan"], "core_root_review": plan["core_root_review"], "core_summary": plan["core_summary"],
        "physical_plan": plan["physical_plan"], "physical_summary": plan["physical_summary"], "source_sha256": pins,
        "output": str(host_path(plan["output"]))}
    return {"schema": "geobc_elastic_coarse_five_extension_v1", **science, "science_sha256": digest(science),
        "base_gate": {"eligible": True, "accepted_true_converged_baselines": 5, "reused_coarse_PASS_cases": 2},
        "array_or_weight_reads_in_request_contract": 0, "core_request_contract_calls": 0}, plan


def selection_gate(cases):
    converged = sum(item.get("numerically_converged") is True and finite(item.get("true_relative_residual"))
                    and item["true_relative_residual"] <= 1e-8 for item in cases.values())
    same = sum(item.get("same_solution_passed") is True for item in cases.values())
    cost_complete = len(cases) == 5 and all(item.get("costs", {}).get("components_complete") is True for item in cases.values())
    beneficial = sum(isinstance(item.get("costs", {}).get("cold_cost_reduction_fraction"), (int, float))
                     and math.isfinite(item["costs"]["cold_cost_reduction_fraction"])
                     and item["costs"]["cold_cost_reduction_fraction"] >= .25 for item in cases.values())
    no_gt = len(cases) == 5 and all(closed(item["read_receipt"]) for item in cases.values())
    passed = converged == same == beneficial == 5 and cost_complete and no_gt
    return {"passed": passed, "inventory_denominator": 5, "numerically_converged_cases": converged,
        "same_solution_passed_cases": same, "cases_with_at_least25percent_cold_cost_reduction": beneficial,
        "complete_cost_components": cost_complete, "target_and_weight_reads_zero": no_gt,
        "minimum_required_gain_fraction": .25, "old_two_costs_reused_without_wrapper_overhead": True,
        "decision": "existing_five_macro_cost_screen_passed_next_scope_requires_root_decision" if passed else "close_exact_fixed32_existing_five_candidate_no_rescue"}


def write_report(output, summary):
    _, _, base = helpers()
    small = {key: summary[key] for key in ("kind", "case_keys", "reused_case_keys", "new_case_keys", "case_provenance",
             "selection_gate", "read_receipt", "parent_read_receipt", "campaign_read_receipt", "costs", "counts")}
    small["cases"] = {key: {name: summary["cases"][key].get(name) for name in ("config", "numerical_state",
        "true_relative_residual", "same_solution_passed", "numerical_comparison", "costs", "counts", "memory")}
        for key in CASES}
    base.write_json(output/"metrics.json", small)
    verdict = "现有五例全部达到完整cold至少25%收益、真残差1e-8与原节点同解。" if summary["selection_gate"]["passed"] else "未通过固定五例全部25%完整cold收益/真收敛/同解门；关闭这一个冻结候选。"
    lines = ["# 现有五CAD固定32步粗空间成本筛选", "", verdict, "",
             "| CAD | 原zero cold秒 | coarse cold秒 | 成本降低 |", "|---|---:|---:|---:|"]
    for key in CASES:
        costs = summary["cases"][key]["costs"]
        def fmt(value, percent=False):
            return "—" if value is None else (f"{value*100:.2f}%" if percent else f"{value:.3f}")
        suffix = "（原2复用）" if key in REUSED else ""
        lines.append(f"| {key}{suffix} | {fmt(costs.get('zero_cold_without_GT_seconds'))} | {fmt(costs.get('coarse_cold_accounted_wall_seconds'))} | {fmt(costs.get('cold_cost_reduction_fraction'), True)} |")
    costs = summary["costs"]
    lines += ["", f"原两例实际stage {costs['parent_actual_stage_wall_seconds']:.3f}秒；本扩展新增累计{costs['cumulative_new_extension_wall_seconds']:.3f}秒；campaign总{costs['cumulative_campaign_wall_seconds']:.3f}/900秒。",
        "只新执行101_56/ver、101_78/ver、102_38/ver；原两例仅小JSON复用，保留原config、成本和来源hash，不重读其预测数组。",
        "新阶段数值reference读取与父campaign读取分列；它们是同方程收敛物理解，不是GT。GT/应力/旧final/封存新test/神经权重均未读。",
        "单例cold为复用但计价的fine setup加实际case wall/原对应stage元数据；不是fresh组合cold重测，不给原两例追加wrapper开销。",
        "同fine curved-P2方程、约束与RBE3载荷，只改预条件。P2→P1不是h粗化，fixed中点掩码可能不nested；不是全谱SPD或工程资格证明。",
        "完整TET10/真实约束物理接口与点云/25特征学习接口信息不同，不和旧final神经误差拼最终solver排名。"]
    path = output/"BRIEF.md"; temporary = path.with_name(path.name+".tmp")
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8"); temporary.replace(path)


def run(plan_path=PLAN, output_path=None):
    began = time.monotonic(); contract, plan = request_contract(plan_path)
    require(plan.get("scientific_rules_frozen") is True and plan.get("actual_execution_released") is True,
            "Root actual extension release required before scientific work")
    review = read_json(plan["root_review"])
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"][SOURCE]
            and review.get("science_recipe_sha256") == digest(contract["plan_science"])
            and review.get("base_evidence_decision") == "extend_existing_five_reuse_two_run_three",
            "Authentic wrapper-scope/provenance/campaign/root release required")
    core, cross, base = helpers()
    output = host_path(output_path or plan["output"])
    require(output == host_path(plan["output"]), "Extension output differs from frozen destination")
    config = {"contract": contract, "stage": KIND}
    if (output/"config.json").exists():
        require(read_json(output/"config.json") == config, "Existing extension immutable config differs")
    if (output/"summary.json").exists():
        previous = read_json(output/"summary.json")
        require(previous["config"] == config, "Existing extension result identity differs")
        if previous["status"] == "completed":
            return previous
    output.mkdir(parents=True, exist_ok=True); base.write_json(output/"config.json", config)
    clock = cross.StageClock(output, config, began)
    parent_seconds = plan["campaign_budget"]["parent_actual_seconds"]
    prior_before_parent_floor = clock.prior
    clock.prior = max(clock.prior, parent_seconds)  # Parent counted exactly once, never added on each resume.
    clock.account("running")
    metadata_path = output/"stage_metadata_receipt.json"
    old_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if old_metadata:
        require(old_metadata["config_sha256"] == digest(config), "Extension metadata receipt belongs to another config")
    actual_metadata = time.monotonic()-began
    metadata_seconds = old_metadata.get("first_metadata_seconds", actual_metadata)
    base.write_json(metadata_path, {"config_sha256": digest(config), "first_metadata_seconds": metadata_seconds,
        "cumulative_actual_metadata_seconds": old_metadata.get("cumulative_actual_metadata_seconds", 0.)+actual_metadata,
        "parent_actual_stage_wall_seconds": parent_seconds, "prior_before_parent_floor_this_call": prior_before_parent_floor,
        "campaign_prior_this_call": clock.prior, "parent_counted_once_policy": "max(ownprior,parentactual)"})
    cases, provenance = {}, {}
    current_new_key, current_new_output = None, None
    try:
        for key in REUSED:
            source = plan["reused_cases"][key]
            item = pinned_json(source)
            require(item["config"] == contract["reused_sources"][key]["original_config"], "Original reused config changed")
            cases[key] = item  # Entire original dict unchanged; do not present it as a wrapper-generated case.
            provenance[key] = {"role": "reused_original_actual_core_case_metadata_only", "source_path": source["path"],
                "source_sha256": source["sha256"], "original_config_preserved": True,
                "arrays_reopened_this_extension": False, "new_GPU_case_execution": False,
                "original_cold_cost_retained_without_wrapper_surcharge": True}
        for key in NEW:
            record = contract["records"][key]; case_output = host_path(record["case_output"])
            current_new_key, current_new_output = key, case_output
            case_cfg = core.case_config(record, contract)
            summary_path = case_output/"summary.json"
            if summary_path.exists():
                item = read_json(summary_path)
                require(item.get("config") == case_cfg, "Saved new-extension case identity differs")
                if item.get("status") == "completed":
                    cases[key] = item
                    provenance[key] = {"role": "new_extension_actual_case_committed_resume_metadata", "source_path": str(summary_path),
                                       "source_sha256": sha(summary_path), "core_source_sha256": contract["source_sha256"][CORE_SOURCE]}
                    clock.account("new_case_reused_after_resume", key); continue
            if clock.remaining() <= 0 and not (case_output/"checkpoint.json").exists() and not (case_output/"endpoint_no_GT.json").exists():
                case_output.mkdir(parents=True, exist_ok=True)
                item = {**core.case_base(record, case_cfg), "numerical_state": "campaign_budget_unattempted_negative",
                        "reason": "No new case after cumulative900s including parent; fixed five denominator retained, no arrays/GT"}
                base.write_json(case_output/"config.json", case_cfg); base.write_json(summary_path, item)
            else:
                # The only scientific dispatch in this wrapper; never core.run or core.request_contract.
                item = core.coarse_case(record, contract, case_output, clock, metadata_seconds)
            cases[key] = item
            provenance[key] = {"role": "new_extension_actual_case" if item["attempted"] else "new_extension_fixed_inventory_unattempted_negative",
                "source_path": str(summary_path), "source_sha256": sha(summary_path),
                "core_source_sha256": contract["source_sha256"][CORE_SOURCE], "original_baseline_not_rerun": True}
            clock.account("new_case_completed", key)
        ordered = {key: cases[key] for key in CASES}
        gate = selection_gate(ordered)
        new_receipt = core.zero_receipt()
        new_receipt["numerical_reference_reads"] = sum(cases[key]["read_receipt"]["numerical_reference_reads"] for key in NEW)
        parent_receipt = read_json(plan["core_summary"]["path"])["read_receipt"]
        campaign_receipt = {name: parent_receipt.get(name, 0)+new_receipt.get(name, 0) for name in new_receipt}
        elapsed = clock.elapsed()
        new_attempted = sum(cases[key]["attempted"] for key in NEW)
        summary = {"status": "completed", "kind": KIND, "config": config, "case_keys": list(CASES), "cases": ordered,
            "reused_case_keys": list(REUSED), "new_case_keys": list(NEW), "reused_cases": 2, "newly_attempted_cases": new_attempted,
            "case_provenance": provenance, "selection_gate": gate, "read_receipt": new_receipt,
            "parent_read_receipt": parent_receipt, "campaign_read_receipt": campaign_receipt,
            "elapsed_seconds": elapsed,
            "costs": {"cumulative_stage_wall_seconds": elapsed, "cumulative_campaign_wall_seconds": elapsed,
                "parent_actual_stage_wall_seconds": parent_seconds, "cumulative_new_extension_wall_seconds": elapsed-parent_seconds,
                "this_call_extension_wall_seconds": time.monotonic()-began, "maximum_stage_wall_seconds": 900,
                "campaign_budget_overshoot_seconds": max(0., elapsed-900), "parent_counted_once": True,
                "new_single_query_metadata_seconds": metadata_seconds, "old_two_costs_or_configs_modified": False,
                "actual_new_metadata_seconds_including_resumes": read_json(metadata_path)["cumulative_actual_metadata_seconds"]},
            "counts": {"fixed_cases": 5, "reused_cases": 2, "new_case_inventory": 3, "newly_attempted_cases": new_attempted,
                "attempted_campaign_cases": 2+new_attempted, "numerically_converged": gate["numerically_converged_cases"],
                "same_solution": gate["same_solution_passed_cases"], "new_numerical_reference_reads": new_receipt["numerical_reference_reads"],
                "parent_numerical_reference_reads": parent_receipt["numerical_reference_reads"],
                "campaign_numerical_reference_reads": campaign_receipt["numerical_reference_reads"],
                "new_setup_Galerkin_fine_K_calls": sum(cases[key].get("counts", {}).get("setup_K_applications", 0) for key in NEW),
                "parent_setup_Galerkin_fine_K_calls": sum(cases[key]["counts"]["setup_K_applications"] for key in REUSED),
                "inherited_eight_fine_K_guard_repeats": 0},
            "scientific_outcome": "frozen_existing_five_macro_speed_gate_passed" if gate["passed"] else "frozen_existing_five_macro_speed_gate_failed",
            "scope": "Existing fixed5 TRAIN/DEV/ver, metadata reuse2 plus only actual new3; no GT or finalsolver claim",
            "interface_comparison_limit": "FullTET10/trueconstraints physics vs pointcloud25 learning differs; no oldfinal neural ranking",
            "old_GT_accuracy_assigned_to_new_endpoint": False, "new_training_updates": 0,
            "parent_outputs_immutable": True, "core_source_immutable": True}
        base.write_json(output/"summary.json", summary); write_report(output, summary); clock.account("completed")
        return summary
    except Exception as error:
        clock.account("failed")
        failed_receipt = core.zero_receipt()
        failed_receipt["numerical_reference_reads"] = sum(cases[key]["read_receipt"]["numerical_reference_reads"]
                                                         for key in NEW if key in cases)
        if current_new_key not in cases and current_new_output is not None and (current_new_output/"ledger.json").exists():
            journal = read_json(current_new_output/"ledger.json")
            require(journal["config_sha256"] == digest(core.case_config(contract["records"][current_new_key], contract)),
                    "Failed new-case receipt belongs to another immutable case")
            failed_receipt["numerical_reference_reads"] += journal["read_receipt"]["numerical_reference_reads"]
        parent_receipt = read_json(plan["core_summary"]["path"])["read_receipt"]
        campaign_receipt = {name: parent_receipt.get(name, 0)+failed_receipt.get(name, 0) for name in failed_receipt}
        base.write_json(output/"summary.json", {"status": "failed", "kind": KIND, "config": config, "case_keys": list(CASES),
            "cases": cases, "case_provenance": provenance, "reused_case_keys": list(REUSED), "new_case_keys": list(NEW),
            "read_receipt": failed_receipt, "parent_read_receipt": parent_receipt, "campaign_read_receipt": campaign_receipt,
            "reason": f"{type(error).__name__}: {error}",
            "costs": {"cumulative_stage_wall_seconds": clock.elapsed(), "parent_actual_stage_wall_seconds": parent_seconds,
                      "cumulative_new_extension_wall_seconds": clock.elapsed()-parent_seconds, "parent_counted_once": True}})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=Path(PLAN))
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.plan, args.output)
