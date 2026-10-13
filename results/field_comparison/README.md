# Saved-model field comparison

Fixed first final CAD 21_119, registered seed 2026. The reference Google Slides field-page layout and palette are followed: FEM truth, prediction and error columns; white background, Arial text, navy title, pink L2 label, turbo field maps and magma error maps with vertical bars. Two rows show ZERO and FIELDWISE. Separate displacement and stress figures preserve the reference 16:9 slide format.

Displacement error is the vector-difference norm divided by full-node true vector RMS; stress error is absolute signed-scalar difference divided by full-node true stress RMS. Both are shown in percent on a common 0–50% scale, with larger errors saturated, as in the reference figure. Full-node L2 metrics are never clipped. Field scales are shared between truth and both models; original boundary midnodes are retained, without enlarged deformation.

All three canonical loads have separate displacement/stress PNGs, two-page PDFs and combined PNG overviews. No model inference, training or final-set tuning was performed. The revealed labels were read solely for the requested visualization.

| Load | Displacement | Stress | Two-page PDF |
|---|---|---|---|
| Vertical | [PNG](ver_displacement.png) | [PNG](ver_stress.png) | [PDF](ver_comparison.pdf) |
| Horizontal | [PNG](hor_displacement.png) | [PNG](hor_stress.png) | [PDF](hor_comparison.pdf) |
| Torsion | [PNG](tor_displacement.png) | [PNG](tor_stress.png) | [PDF](tor_comparison.pdf) |
