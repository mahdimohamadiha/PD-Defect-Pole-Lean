# Phase 1b: visual error analysis (tuned DINOv2 `cls` + `mt`, test split)

Contact sheets: [`results/analysis/phase1/error_sheets/`](../results/analysis/phase1/error_sheets/). Each caption
gives the file name, the true label (T=) and the predictions of the 6 runs (3 seeds of `cls`, then 3 of `mt`).
An image is listed when at least 3 of the 6 runs got it wrong. Images: PD-Defect, CC BY 4.0.

## What `Rejected` contains (random train sample)
Mixed reasons, with no single visual definition. The sample shows poles too distant or small, several poles or
transformer structures, close-ups where the pole is partly out of frame, night or backlit shots, landscape-format
shots and scenes with no clear pole. Train statistics:

| class | n | dark images (mean grey < 70) | landscape format | has a keypoint label | ≥ 2 annotated poles (of those with labels) |
|---|---|---|---|---|---|
| Leaned | 464 | 0.4% | 1.1% | 91% | 12.8% |
| Straight | 962 | 0.6% | 0.1% | 85% | 10.8% |
| Rejected | 280 | **6.8%** | **5.0%** | **12.5%** | 14.3% |

## Error patterns
1. **Darkness acts as a shortcut for `Rejected`.** 4 of the 7 images that are wrongly predicted `Rejected` are dusk or
   night shots of a valid single pole (1024, 1460, 1560, 933). Dark images are about 10 times more frequent in
   train `Rejected`.
2. **The `Rejected` labels themselves are ambiguous.** Several true-`Rejected` images that all models miss show one
   clear pole (2389, 2402, 799). Others show two or more poles (2032, 543) or a pole that is cut off (1795).
3. **Multi-pole scenes leave the target unclear for Leaned vs Straight.** In 893 and 1544 (T = Straight) a strongly
   tilted stay pole or strut stands next to a straight pole, and the models predict Leaned. In 190 and 790 several
   poles are visible.
4. **Camera tilt and perspective.** Several true-`Leaned` poles predicted Straight (100, 127, 1267, 1290) are almost
   vertical in the image. Annotators probably judged the lean against scene verticals such as building edges, while
   the image axis gives no signal.
5. These cases match phase 1a: the Leaned/Straight errors sit in the 2–6° band of the GT image-axis angle.

## Implications
- Leaned vs Straight is close to the label-consistency ceiling (90.4% vs 91.6%). The remaining errors are mostly
  label or definition ambiguity (target pole, scene vs image vertical), which architecture changes cannot fix.
- `Rejected` is the main source of Macro-F1 loss, and part of that loss is a darkness shortcut. This is a concrete,
  testable target.
