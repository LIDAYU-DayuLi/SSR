# Lazy-dispatch metric entrypoint to avoid circular imports and missing symbols.

def _import(name):
    mod = __import__(name, fromlist=['*'])
    return mod

# Try import common metrics if they exist in this repo
try:
    _ps = _import('basicsr.metrics.psnr_ssim')
    calculate_psnr = getattr(_ps, 'calculate_psnr', None)
    calculate_ssim = getattr(_ps, 'calculate_ssim', None)
except Exception:
    calculate_psnr = None
    calculate_ssim = None

# Optional LPIPS
try:
    _lp = _import('basicsr.metrics.lpips')
    calculate_lpips = getattr(_lp, 'calculate_lpips', None)
except Exception:
    calculate_lpips = None

def calculate_metric(metric_opt, img, img2):
    """
    Unified metric API used by some repos.
    metric_opt: dict with keys like:
      - type: 'psnr' | 'ssim' | 'lpips' (case-insensitive)
      - crop_border: int (optional)
      - test_y_channel: bool (optional, for PSNR/SSIM)
      - better: additional kwargs passed through
    img, img2: tensors or ndarrays as expected by underlying metric impl
    """
    name = (metric_opt.get('type', 'psnr') if isinstance(metric_opt, dict) else str(metric_opt)).lower()
    kw = metric_opt.copy() if isinstance(metric_opt, dict) else {}
    crop_border = kw.pop('crop_border', 0)
    test_y = kw.pop('test_y_channel', False)

    if name == 'psnr':
        if calculate_psnr is None:
            raise ImportError("calculate_psnr is not available in basicsr.metrics.psnr_ssim")
        return calculate_psnr(img, img2, crop_border=crop_border, test_y_channel=test_y, **kw)

    if name == 'ssim':
        if calculate_ssim is None:
            raise ImportError("calculate_ssim is not available in basicsr.metrics.psnr_ssim")
        return calculate_ssim(img, img2, crop_border=crop_border, test_y_channel=test_y, **kw)

    if name == 'lpips':
        if calculate_lpips is None:
            raise ImportError("calculate_lpips is not available in basicsr.metrics.lpips")
        return calculate_lpips(img, img2, **kw)

    # Fallback: try to import a module with the same name and call calculate_<name>
    try:
        mod = _import(f'basicsr.metrics.{name}')
        fn = getattr(mod, f'calculate_{name}')
        return fn(img, img2, **kw)
    except Exception as e:
        raise ImportError(f"Unknown metric type '{name}', and no basicsr.metrics.{name} module found") from e

__all__ = ['calculate_metric']
