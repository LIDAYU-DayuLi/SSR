<h1 align="center">[ECCV 2026] Pixel Ignores, Superpixel Sees: Adverse Weather Image Restoration via Semantic-Center SSM</h1>

<p align="center">Official PyTorch implementation</p>

<p align="center">
  <a href="https://github.com/LIDAYU-DayuLi/SSR"><img src="https://img.shields.io/badge/GitHub-SSR-blue" alt="GitHub"></a>
  <a href="https://github.com/LIDAYU-DayuLi/SSR/issues"><img src="https://img.shields.io/github/issues/LIDAYU-DayuLi/SSR" alt="Issues"></a>
</p>

<p align="center">
  <b>Pixel Ignores, Superpixel Sees: Adverse Weather Image Restoration via Semantic-Center SSM</b><br>
  ECCV 2026
</p>

| Motivation | Model Architecture |
| :--: | :--: |
| <img src="assets/figures/motivation.png" width="480"/> | <img src="assets/figures/architecture.png" width="480"/> |

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
