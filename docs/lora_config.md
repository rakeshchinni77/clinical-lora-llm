# LoRA Configuration Rationale

## Required Statement

“Due to hardware constraints, I used FLAN-T5-base. The pipeline is fully compatible with larger models like Mistral-7B when deployed on GPU infrastructure.”

## r (LoRA rank)

- Configured value: `r=16`
- Why: balances adaptation capacity and memory footprint for constrained hardware.
- Trade-off: increasing `r` can improve expressivity but raises compute and overfitting risk.

## lora_alpha

- Configured value: `lora_alpha=32`
- Why: scales adapter updates to provide meaningful task adaptation at low parameter cost.
- Trade-off: too high can destabilize updates, too low can underfit.

## target_modules

- Configured value: `q,v`
- Why: adapting attention query/value projections is a strong parameter-efficient baseline.
- Trade-off: expanding target modules may improve quality but increases training cost.

## Trade-offs and Future Tuning

- Current setup prioritizes reproducibility and resource efficiency over maximum model capacity.
- Future tuning directions:
  1. Try `r=32`, `lora_alpha=64` with the same evaluation protocol.
  2. Evaluate broader `target_modules` on GPU infrastructure.
  3. Add factuality-aware decoding and post-generation checks to reduce hallucination risk.
