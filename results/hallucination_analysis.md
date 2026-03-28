# Hallucination Analysis

This document captures potential hallucinations observed in generated summaries from the fine-tuned LoRA model. Examples are taken from `results/evaluation_metrics.json` sample predictions.

## Case 1 (Example ID: test-0)

### Source Text

The source discusses NICU parent empowerment, maternal stress/anxiety, and the COPE program in Iranian hospitals.

### Generated Summary (Fine-Tuned)

"nicus have been established in iran since about 1984 ... mothers can also visit their infants in specific hours, but they can not stay there and no health care is provided for them ..."

### Highlighted Hallucination

Potential unsupported detail: **"no health care is provided for them"**.

### Cause Hypothesis

The model appears to over-copy and over-compress long context windows, then amplifies a local sentence into a broad claim without enough summarization control.

## Case 2 (Example ID: test-1)

### Source Text

The source discusses regulation of cognitive enhancement devices (CEDs), treatment vs enhancement framing, and implications for EU medical device regulation.

### Generated Summary (Fine-Tuned)

"non - therapeutic devices, which do not aim to improve health problems, would not be medical devices on the mdd definition ..."

### Highlighted Hallucination

Potential fabrication/over-assertion: **"would not be medical devices"** as a definitive regulatory conclusion.

### Cause Hypothesis

Regulatory language in the source is nuanced. The model likely collapses conditional/legal nuance into a categorical claim due to exposure bias during autoregressive decoding.

## Case 3 (Example ID: test-2)

### Source Text

The source describes Albanian primary healthcare quality assessment and physician perception data in a cross-sectional study.

### Generated Summary (Fine-Tuned)

"quality of primary health care services in albania is a major issue in developing and transitional countries ..." (repetitive continuation).

### Highlighted Hallucination

Potential unsupported extrapolation: **"major issue in developing and transitional countries"** generalized beyond study scope.

### Cause Hypothesis

The model shows discourse-level drift and repetition under long generation, suggesting insufficient penalty controls (e.g., repetition penalty, tighter max target length) and limited factual grounding.

## Cross-Case Pattern Summary

- Hallucinations are mostly over-generalization, over-assertion, or scope drift.
- Long source inputs increase risk of copying broad claims without clear evidence boundaries.
- Factuality can be improved with stricter decoding settings, retrieval grounding, and explicit factual consistency checks.
