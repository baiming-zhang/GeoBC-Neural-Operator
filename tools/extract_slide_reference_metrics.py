"""Persist manual readings of four native slide thumbnails and their notes.

This is a transcription, not an OCR or a reconstruction of missing run files.
The corresponding 1600x900 thumbnails were inspected with view_image.
"""
import json
from pathlib import Path


def main():
    metrics = []
    def record(slide, case, method, budget, percent, source_kind="printed_chart_label", decimals=1, **extra):
        metrics.append({"slide": slide, "slide_object_id": f"p{slide}", "case": case,
                        "method": method, "budget": budget,
                        "metric": "decoded source signed stress relative L2", "value_percent": percent,
                        "value_relative_l2": percent / 100,
                        "source_kind": source_kind, "read_precision_percent_decimal_places": decimals,
                        "read_precision_status": "Manual reading of visible rounded slide label; not a source run artifact.",
                        "source_artifact_available": False, **extra})

    for case, value in [("506_533/ver", 45.0), ("506_25/tor", 78.4)]:
        record(4, case, "TS3 short diagnostic", {"evaluation_update": 800, "stopped_update_from_notes": 897}, value)
    for case, value in [("506_533/ver", 5.5), ("506_25/tor", 11.5)]:
        record(4, case, "TS3 long", {"epochs": 5000, "updates": 10000,
                                   "best_update_from_notes": 9900}, value)
    for method, values in [("Transolver++ direct linear", [90.3, 99.0]),
                           ("Transolver++ direct asinh", [70.3, 100.0])]:
        for case, value in zip(["506_533/ver", "506_25/tor"], values):
            record(5, case, method, {"epochs": 500, "updates": 1000, "learning_rate": .003,
                                    "checkpoint_selection_from_notes": "decoded physical normalized MSE"}, value)
    for method in ["Small NGP reference", "NGP + TS3 correction", "NGP + Transolver++ correction"]:
        for case, value in [("506_533/ver", 8.5), ("506_25/tor", 12.9)]:
            budget = ({"updates": 1000, "budget_source": "slide 9 notes for the same ngp_stress reference"}
                      if method == "Small NGP reference" else {"epochs": 500, "learning_rate": .003,
                                                               "budget_source": "slide 6 notes; correction budget only"})
            record(6, case, method, budget, value,
                   limitation="Frozen NGP field was added only at output and was not an input to the correction network.",
                   comparison_precision="Identical rounded labels do not establish identical underlying errors.")
    record(17, "50 training CADs / mean per case", "Shared large hash grid before continuation",
           {"sampled_updates": 10000, "global_batch_nodes": 51200,
            "equivalent_full_passes_from_notes": 45.97}, 23.70,
           source_kind="printed_slide_footer", decimals=2)
    record(17, "50 training CADs / mean per case", "Shared large hash grid after continuation",
           {"true_epochs": 1000, "updates": 218000, "global_batch_nodes": 51200,
            "best_epoch_from_notes": 990, "best_update_from_notes": 215820,
            "optimizer_from_notes": "fresh Adam from the parent's best model weights"}, 15.11,
           source_kind="printed_slide_footer", decimals=2,
           checkpoint_note="Footer says after 1000 epochs. Notes select epoch990 by worst-case L2; do not silently call this the epoch1000 checkpoint.")

    curve = []
    for epoch, mean, worst in [(1, 27.0, 39.0), (50, 23.0, 35.5), (1000, 15.0, 28.5)]:
        for measure, value in [("mean per case", mean), ("worst case", worst)]:
            curve.append({"slide": 17, "case": f"50 training CADs / {measure}",
                          "method": "Shared large hash grid continuation", "budget": {"true_epoch": epoch},
                          "metric": "decoded source signed stress relative L2", "value_percent_approximate": value,
                          "read_precision_status": "Approximate visual read from an unlabeled curve; about 0.5 percentage-point precision. Not a source run artifact.",
                          "source_kind": "unlabeled_curve_visual_read", "source_artifact_available": False})

    notes = []
    for case, u, stress in [("506_533/ver", .5768, 5.5437), ("506_25/tor", 2.2882, 11.5111)]:
        for field, value in [("displacement vector relative L2", u), ("source signed stress relative L2", stress)]:
            notes.append({"slide": 4, "case": case, "method": "TS3 long",
                          "budget": {"epochs": 5000, "updates": 10000, "best_update": 9900},
                          "metric": field, "value_percent": value,
                          "source_kind": "speaker_notes_claim", "source_artifact_available": False,
                          "read_precision_status": "Literal number in speaker notes; not independently recovered from the missing summary.json."})
    for case, value in [("506_533/ver", .390), ("506_25/tor", 1.037)]:
        notes.append({"slide": 6, "case": case, "method": "NGP displacement-only reference",
                      "budget": {"updates": None, "epochs": None, "combined_elapsed_seconds_approximate": 10.2},
                      "metric": "displacement vector relative L2", "value_percent": value,
                      "source_kind": "speaker_notes_claim", "source_artifact_available": False,
                      "read_precision_status": "Literal number in speaker notes; u-only training budget is not specified."})

    report = {"presentation_url": "https://docs.google.com/presentation/d/1g5S0QCZSDR_igYFWFe20SQ3nw2yvgYbXk_t4QSmKpBU/edit",
              "slides_inspected": [4, 5, 6, 17], "thumbnail_size": [1600, 900],
              "reading_method": "Fresh google_drive_get_slide_thumbnail LARGE; returned native image pointers plus documented direct-fetch contentUrl fallback; viewed the actual PNG pixels.",
              "evidence_images": {str(slide): f"references/thumbnail_reading/p{slide}.png" for slide in [4, 5, 6, 17]},
              "scope": "Reference result transcription only. Every value is a slide-chart/manual reading or a notes claim, not an available original numerical source artifact.",
              "metrics": metrics, "approximate_curve_points": curve, "note_metrics": notes,
              "architecture_parameters": {
                  "extra_parameters_visible_in_four_images": None,
                  "Transolver++_from_slide5_notes": {"layers": 16, "width": 256, "official_backbone_unchanged": True},
                  "TS3": "No depth/width/slice parameter is printed in these four charts.",
                  "NGP": "No grid table/resolution/MLP parameters are printed in these four charts.",
                  "pooled50_hashgrid": "No architecture parameters are printed in slide17's curve image; use the other slide notes, without claiming a source artifact."},
              "reference_50_case_gap": "The plotted 50 are selected from the unavailable bespoke 270-CAD train pool. Our official archive-prefix 50 is a documented substitute, so these are separate data populations."}
    Path("notes/slide_reference_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {len(metrics)} printed labels, {len(curve)} approximate curve points, {len(notes)} separate speaker-note claims.")


if __name__ == "__main__":
    main()
