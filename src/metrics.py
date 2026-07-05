from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if mask.any() else np.nan


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else np.nan


def pcc(y_true, y_pred):
    a, b = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def r2_score_np(y_true, y_pred):
    a, b = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return np.nan
    a, b = a[mask], b[mask]
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def spatial_corr_loss_np(y_true, y_pred, eps: float = 1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    B, T, H, W = y_true.shape
    yt = y_true.reshape(B * T, -1)
    yp = y_pred.reshape(B * T, -1)
    yt = yt - yt.mean(axis=1, keepdims=True)
    yp = yp - yp.mean(axis=1, keepdims=True)
    num = np.sum(yt * yp, axis=1)
    den = np.sqrt((np.sum(yt ** 2, axis=1) + eps) * (np.sum(yp ** 2, axis=1) + eps))
    return float(np.mean(1.0 - num / den))


def acc(y_true, y_pred):
    """Mean spatial anomaly correlation coefficient."""
    return 1.0 - spatial_corr_loss_np(y_true, y_pred)


def _global_ssim(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    data_range = max(float(np.nanmax(gt)), float(np.nanmax(pred))) - min(float(np.nanmin(gt)), float(np.nanmin(pred)))
    if not np.isfinite(data_range) or data_range <= 1e-12:
        return 1.0 if np.allclose(gt, pred) else 0.0
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mux, muy = np.nanmean(gt), np.nanmean(pred)
    vx, vy = np.nanmean((gt - mux) ** 2), np.nanmean((pred - muy) ** 2)
    cov = np.nanmean((gt - mux) * (pred - muy))
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))


def ssim_map(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    data_range = max(float(np.nanmax(gt)), float(np.nanmax(pred))) - min(float(np.nanmin(gt)), float(np.nanmin(pred)))
    if not np.isfinite(data_range) or data_range <= 1e-12:
        return 1.0 if np.allclose(gt, pred) else 0.0
    min_dim = min(gt.shape[-2], gt.shape[-1])
    if skimage_ssim is not None and min_dim >= 3:
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        try:
            return float(skimage_ssim(gt, pred, data_range=data_range, win_size=win_size))
        except Exception:
            pass
    return _global_ssim(gt, pred)


def mean_ssim(y_true, y_pred, max_maps: int = 2048):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    maps_true = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
    maps_pred = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
    n = maps_true.shape[0]
    idx = np.linspace(0, n - 1, max_maps).astype(int) if n > max_maps else np.arange(n)
    return float(np.nanmean([ssim_map(maps_true[i], maps_pred[i]) for i in idx]))


def all_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "PCC": pcc(y_true, y_pred),
        "R2": r2_score_np(y_true, y_pred),
        "SSIM": mean_ssim(y_true, y_pred),
        "ACC": acc(y_true, y_pred),
    }
