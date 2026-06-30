# SSR

Code for SSR (ECCV 2026).

## Project Structure

- `basicsr/` — Training and testing framework (based on BasicSR)
- `Allweather/` — All-weather restoration configs and utilities
- `basicsr/models/archs/SSR_arch.py` — SSR network architecture

## Requirements

- Python 3.8+
- PyTorch
- einops
- mamba-ssm
- tensorboardX

## Training

```bash
cd basicsr
python train.py -opt ../Allweather/Options/Allweather_SSR.yml
```

Set `dataroot_gt` and `dataroot_lq` in the config file before training.

## Testing

```bash
cd basicsr
python test.py -opt ../Allweather/Options/Allweather_SSR.yml
```
