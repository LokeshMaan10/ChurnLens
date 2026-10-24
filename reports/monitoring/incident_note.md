# Incident note — drift A → B
_Monitoring agent (offline mode), from the drift report generated 2026-10-24T23:07:09._

**Prediction drift:** PSI 0.019 (stable). Mean score 6.42% → 6.68%.
**Features:** 76 monitored; 3 flagged (1 major, 2 moderate, 0 missing-share flags).

## Data issue (0)
- none

## Structural (clock) shift (1)
- `days_since_first_transaction`: PSI 0.687 (major), missing 0.2% → 0.2%, median 510 → 528
  For customers present in both snapshots the value moved by 28 days (median), the 28-day gap between cutoffs: it tracks the calendar, not behaviour. If the model relies on it, cap it or measure it from a fixed point, then retrain.

## Likely calendar / seasonal (0)
- none

## Behaviour shift (2)
- `days_since_last_transaction`: PSI 0.185 (moderate), missing 0.2% → 0.2%, median 15 → 15
- `days_to_expiry`: PSI 0.104 (moderate), missing 0.2% → 0.3%, median 17 → 16

Calendar context: the reference window overlaps Lunar New Year holiday (Taiwan); short-window listening features are expected to differ for that reason (a timing coincidence, not a tested cause).

## Retraining decision
**RETRAIN** — calibration gap -0.0241 (mean predicted - observed) exceeds ±0.02.

## Next steps
- Data issues: check the pipeline/source for the flagged columns before trusting scores that depend on them.
- Behaviour shifts: confirm with fresh labels (calibration gap, PR-AUC) per the retraining policy in `configs/monitoring.yaml`.
