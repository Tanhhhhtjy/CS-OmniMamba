# CS-OmniMamba

Training pipeline for short-term precipitation prediction using PWV, radar, and rain targets.

## Dataset

Expected layout:

```
data/
  PWV/
  RADAR/
  RAIN/
```

File names are timestamps, for example `2023-08-01-00-00-00.png`.

Matching rule:
- PWV timestamp t
- nearest RADAR within 1 hour of t
- RAIN targets at t+1h, t+2h, t+3h must exist

## Dependencies

- `mamba-ssm` is required for the model blocks

## Run

```bash
python train.py --data-root ./data
```

Outputs are written to `./results` by default.
