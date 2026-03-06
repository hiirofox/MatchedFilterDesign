import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This implementation requires PyTorch.") from e


def spectral_factorization(coeffs):
    N = len(coeffs) - 1
    poly_z = np.zeros(2 * N + 1)
    poly_z[N] = coeffs[0]
    for i in range(1, N + 1):
        poly_z[N - i] = coeffs[i] / 2.0
        poly_z[N + i] = coeffs[i] / 2.0

    roots = np.roots(poly_z)
    sorted_indices = np.argsort(np.abs(roots))
    selected_roots = roots[sorted_indices[:N]]
    return np.real(np.poly(selected_roots))


def design_matched_iir_from_response(R_target, w_grid, order=4):
    N = order
    num_points = len(w_grid)
    if len(R_target) != num_points:
        raise ValueError("R_target and w_grid must have the same length")

    W = np.ones(num_points)
    for _ in range(6):
        A = np.zeros((num_points, 2 * N + 1))
        A[:, 0] = W
        for k in range(1, N + 1):
            A[:, k] = W * np.cos(k * w_grid)
            A[:, k + N] = -W * R_target * np.cos(k * w_grid)

        x, _, _, _ = np.linalg.lstsq(A, W * R_target, rcond=None)
        d_temp = np.concatenate(([1.0], x[N + 1 : 2 * N + 1]))
        Q_vals = d_temp[0] + sum(d_temp[k] * np.cos(k * w_grid) for k in range(1, N + 1))
        W = 1.0 / (np.abs(Q_vals) + 1e-6)

    c = x[0 : N + 1]
    d = np.concatenate(([1.0], x[N + 1 : 2 * N + 1]))
    b_unscaled = spectral_factorization(c)
    a_final = spectral_factorization(d)

    mean_mag_target = np.mean(np.sqrt(R_target))
    _, h_unscaled = signal.freqz(b_unscaled, a_final, worN=w_grid)
    mean_mag_digital = np.mean(np.abs(h_unscaled))
    gain_correction = mean_mag_target / (mean_mag_digital + 1e-12)
    b_final = b_unscaled * gain_correction
    return b_final, a_final


def _roots_to_section_init(roots, order):
    roots = np.asarray(roots, dtype=np.complex128)
    used = np.zeros(len(roots), dtype=bool)
    pairs = []
    singles = []

    for i, r in enumerate(roots):
        if used[i]:
            continue
        if abs(r.imag) < 1e-8:
            singles.append(float(np.clip(r.real, -0.995, 0.995)))
            used[i] = True
            continue

        target = np.conj(r)
        d = np.abs(roots - target)
        d[used] = np.inf
        j = int(np.argmin(d))
        if i != j and not used[j] and np.isfinite(d[j]):
            used[i] = True
            used[j] = True
            rr = float(np.clip(abs(r), 1e-4, 0.995))
            th = float(np.clip(abs(np.angle(r)), 1e-4, math.pi - 1e-4))
            pairs.append((rr, th))
        else:
            singles.append(float(np.clip(r.real, -0.995, 0.995)))
            used[i] = True

    q = order // 2
    pairs = pairs[:q]
    while len(pairs) < q:
        idx = len(pairs) + 1
        pairs.append((0.5, idx * math.pi / (q + 1)))

    single = 0.0
    if order % 2 == 1:
        single = singles[0] if singles else 0.0
    return pairs, single


def _inv_sigmoid(y):
    y = np.clip(y, 1e-4, 1 - 1e-4)
    return np.log(y / (1 - y))

import math
import numpy as np
import scipy.signal as signal
import torch


def design_matched_iir_from_response_nn(
    R_target,
    w_grid,
    order=4,
    num_steps=600,
    lr=0.05,
    vis_db_focus=-40.0,
    vis_db_soft=1.0,
    vis_floor=0.05,
    slope_weight=0.0,
    relmag_weight=1.0,
    device=None,
):
    """
    specirls warm start + direct-coefficient LBFGS refinement

    Notes
    -----
    1) Stability is NOT enforced here.
    2) Final (b, a) can be post-processed externally (e.g. MATLAB pole reflection).
    3) This directly optimizes polynomial coefficients, which is more aggressive
       than stable pole/zero parameterization.
    """
    if len(R_target) != len(w_grid):
        raise ValueError("R_target and w_grid must have the same length")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_r = torch.float64
    dtype_c = torch.complex128

    R_target = np.asarray(R_target, dtype=np.float64)
    w_grid = np.asarray(w_grid, dtype=np.float64)
    mag_target = np.sqrt(np.maximum(R_target, 0.0))
    mag_target_db = 20.0 * np.log10(mag_target + 1e-12)

    # ------------------------------------------------------------
    # Warm start from spectral-factorization IRLS
    # ------------------------------------------------------------
    b0, a0 = design_matched_iir_from_response(R_target, w_grid, order=order)

    b0 = np.asarray(b0, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)

    # normalize so a[0] = 1
    if abs(a0[0]) < 1e-12:
        a0[0] = 1.0
    b0 = b0 / a0[0]
    a0 = a0 / a0[0]

    # ------------------------------------------------------------
    # Tensors
    # ------------------------------------------------------------
    w_t = torch.tensor(w_grid, dtype=dtype_r, device=device)
    xlog = torch.log(torch.clamp(w_t, min=1e-8))
    zinv = torch.exp((-1j * w_t).to(dtype_c))

    mag_t = torch.tensor(mag_target, dtype=dtype_r, device=device)
    mag_db_t = torch.tensor(mag_target_db, dtype=dtype_r, device=device)

    # direct coefficient parameters
    b_param = torch.nn.Parameter(torch.tensor(b0, dtype=dtype_r, device=device))
    a_free = torch.nn.Parameter(torch.tensor(a0[1:], dtype=dtype_r, device=device))

    params = [b_param, a_free]

    # ------------------------------------------------------------
    # frequency response from direct polynomial coefficients
    # ------------------------------------------------------------
    def response_mag():
        a_full = torch.cat(
            [torch.ones(1, dtype=dtype_r, device=device), a_free], dim=0
        )

        # B(e^jw) = sum b_k z^{-k}
        # A(e^jw) = sum a_k z^{-k}
        B = torch.zeros_like(zinv, dtype=dtype_c)
        A = torch.zeros_like(zinv, dtype=dtype_c)

        zpow = torch.ones_like(zinv, dtype=dtype_c)
        for k in range(len(b_param)):
            if k == 0:
                zpow = torch.ones_like(zinv, dtype=dtype_c)
            else:
                zpow = zpow * zinv
            B = B + b_param[k].to(dtype_c) * zpow

        zpow = torch.ones_like(zinv, dtype=dtype_c)
        for k in range(len(a_full)):
            if k == 0:
                zpow = torch.ones_like(zinv, dtype=dtype_c)
            else:
                zpow = zpow * zinv
            A = A + a_full[k].to(dtype_c) * zpow

        H = B / (A + 1e-12)
        mag = torch.abs(H)
        return mag, a_full

    # ------------------------------------------------------------
    # Visibility weighting
    # ------------------------------------------------------------
    with torch.no_grad():
        w_vis = vis_floor + (1.0 - vis_floor) * torch.sigmoid(
            (mag_db_t - vis_db_focus) / vis_db_soft
        )
        w_vis = w_vis / torch.mean(w_vis)

    # ------------------------------------------------------------
    # LBFGS
    # ------------------------------------------------------------
    opt = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=num_steps,
        max_eval=max(2 * num_steps, 100),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    loss_log = []

    def closure():
        opt.zero_grad()

        mag, a_full = response_mag()
        mag_db = 20.0 * torch.log10(mag + 1e-12)

        err_db = mag_db - mag_db_t
        db_loss = (w_vis * err_db.square()).mean()

        if slope_weight > 0.0:
            derr_db = torch.gradient(err_db, spacing=(xlog,))[0]
            slope_loss = slope_weight * (w_vis * derr_db.square()).mean()
        else:
            slope_loss = torch.zeros((), dtype=dtype_r, device=device)

        if relmag_weight > 0.0:
            relmag_loss = relmag_weight * (
                w_vis * ((mag - mag_t) / (mag_t + 0.05)).square()
            ).mean()
        else:
            relmag_loss = torch.zeros((), dtype=dtype_r, device=device)

        # very light denominator regularization, only to avoid absurd blow-up
        reg_a = 1e-6 * torch.sum(a_full.square())
        reg_b = 1e-8 * torch.sum(b_param.square())

        loss = db_loss + slope_loss + relmag_loss + reg_a + reg_b
        loss.backward()

        loss_log.append(float(loss.detach().cpu()))
        return loss

    opt.step(closure)

    # ------------------------------------------------------------
    # Final coefficients
    # ------------------------------------------------------------
    b_final = b_param.detach().cpu().numpy().astype(np.float64)
    a_final = np.concatenate(
        [[1.0], a_free.detach().cpu().numpy().astype(np.float64)]
    )

    # normalize again
    if abs(a_final[0]) < 1e-12:
        a_final[0] = 1.0
    b_final = b_final / a_final[0]
    a_final = a_final / a_final[0]

    # simple gain correction
    _, h = signal.freqz(b_final, a_final, worN=w_grid)
    gain_correction = np.mean(mag_target) / (np.mean(np.abs(h)) + 1e-12)
    b_final = b_final * gain_correction

    return b_final, a_final



def design_matched_iir_from_response_nn2(
    R_target,
    w_grid,
    order=4,
    num_steps=2200,          # total budget hint; internally split into phases
    lr=0.03,                # Adam warmup lr
    vis_db_focus=-40.0,
    vis_db_soft=4.0,
    vis_floor=0.05,
    slope_weight=1.0,
    relmag_weight=0.0,
    device=None,
):
    """
    Extreme refinement optimizer:
        specirls warm start
        -> short Adam warmup
        -> LBFGS stage 1
        -> LBFGS stage 2
        -> best-so-far selection
        -> accept/reject vs baseline

    Notes
    -----
    1) Directly optimizes polynomial coefficients (b, a[1:]).
    2) Stability is NOT enforced here by design.
       You can reflect poles later in MATLAB if needed.
    3) The optimizer is guarded:
       if refinement is not better than specirls by the internal score,
       it falls back to specirls.
    """
    if len(R_target) != len(w_grid):
        raise ValueError("R_target and w_grid must have the same length")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_r = torch.float64
    dtype_c = torch.complex128

    # ------------------------------------------------------------
    # data prep
    # ------------------------------------------------------------
    R_target = np.asarray(R_target, dtype=np.float64)
    w_grid = np.asarray(w_grid, dtype=np.float64)

    mag_target = np.sqrt(np.maximum(R_target, 0.0))
    mag_target_db = 20.0 * np.log10(mag_target + 1e-12)

    w_t = torch.tensor(w_grid, dtype=dtype_r, device=device)
    xlog = torch.log(torch.clamp(w_t, min=1e-8))
    zinv = torch.exp((-1j * w_t).to(dtype_c))

    mag_t = torch.tensor(mag_target, dtype=dtype_r, device=device)
    mag_db_t = torch.tensor(mag_target_db, dtype=dtype_r, device=device)

    # ------------------------------------------------------------
    # baseline warm start: spectral-factorization IRLS
    # ------------------------------------------------------------
    #b0, a0 = design_matched_iir_from_response(R_target, w_grid, order=order)
    b0 = np.array([1,0,0,0,0], dtype=np.float64)
    a0 = np.array([1,0,0,0,0], dtype=np.float64)

    b0 = np.asarray(b0, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)

    if abs(a0[0]) < 1e-12:
        a0[0] = 1.0
    b0 = b0 / a0[0]
    a0 = a0 / a0[0]

    # direct trainable coefficients
    b_param = torch.nn.Parameter(torch.tensor(b0, dtype=dtype_r, device=device))
    a_free = torch.nn.Parameter(torch.tensor(a0[1:], dtype=dtype_r, device=device))
    params = [b_param, a_free]

    # for stay-close regularization
    b0_t = torch.tensor(b0, dtype=dtype_r, device=device)
    a0_t = torch.tensor(a0, dtype=dtype_r, device=device)

    # ------------------------------------------------------------
    # visibility weighting
    # ------------------------------------------------------------
    with torch.no_grad():
        w_vis = vis_floor + (1.0 - vis_floor) * torch.sigmoid(
            (mag_db_t - vis_db_focus) / max(vis_db_soft, 1e-6)
        )
        w_vis = w_vis / torch.mean(w_vis)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def pack_a_full():
        return torch.cat(
            [torch.ones(1, dtype=dtype_r, device=device), a_free], dim=0
        )

    def response_mag():
        a_full = pack_a_full()

        B = torch.zeros_like(zinv, dtype=dtype_c)
        A = torch.zeros_like(zinv, dtype=dtype_c)

        zpow = torch.ones_like(zinv, dtype=dtype_c)
        for k in range(len(b_param)):
            if k == 0:
                zpow = torch.ones_like(zinv, dtype=dtype_c)
            else:
                zpow = zpow * zinv
            B = B + b_param[k].to(dtype_c) * zpow

        zpow = torch.ones_like(zinv, dtype=dtype_c)
        for k in range(len(a_full)):
            if k == 0:
                zpow = torch.ones_like(zinv, dtype=dtype_c)
            else:
                zpow = zpow * zinv
            A = A + a_full[k].to(dtype_c) * zpow

        H = B / (A + 1e-12)
        mag = torch.abs(H)
        mag_db = 20.0 * torch.log10(mag + 1e-12)
        return mag, mag_db, a_full

    def soft_worst_abs(err_db, tau=12.0):
        # smooth approximation to max / high percentile
        x = tau * torch.abs(err_db)
        x = torch.clamp(x, max=80.0)
        return torch.logsumexp(x, dim=0) / tau - math.log(err_db.numel()) / tau

    def anchor_indices():
        n = len(w_grid)
        idx_lo = max(0, min(n - 1, int(round(0.0008 * (n - 1)))))
        idx_hi = max(0, min(n - 1, int(round(0.9992 * (n - 1)))))
        idx_peak = int(np.argmax(mag_target_db))
        return idx_lo, idx_peak, idx_hi

    idx_lo, idx_peak, idx_hi = anchor_indices()

    def internal_loss(
        stage="adam",
        stay_weight=1e-30,
        slope_w=None,
        relmag_w=None,
        softworst_w=0.00006,
        anchor_w=0.10,
    ):
        if slope_w is None:
            slope_w = slope_weight
        if relmag_w is None:
            relmag_w = relmag_weight

        mag, mag_db, a_full = response_mag()
        err_db = mag_db - mag_db_t

        # main visible-region dB fit
        db_loss = (w_vis * err_db.square()).mean()

        # slope consistency on log-frequency
        if slope_w > 0.0:
            derr_db = torch.gradient(err_db, spacing=(xlog,))[0]
            slope_loss = slope_w * (w_vis * derr_db.square()).mean()
        else:
            slope_loss = torch.zeros((), dtype=dtype_r, device=device)

        # relative linear magnitude
        if relmag_w > 0.0:
            relmag_loss = relmag_w * (
                w_vis * ((mag - mag_t) / (mag_t + 0.05)).square()
            ).mean()
        else:
            relmag_loss = torch.zeros((), dtype=dtype_r, device=device)

        # soft worst-case / high-percentile proxy
        softworst_loss = softworst_w * soft_worst_abs(err_db)

        # anchors: low / peak / high
        anchor_loss = anchor_w * (
            torch.abs(err_db[idx_lo]) +
            1.5 * torch.abs(err_db[idx_peak]) +
            torch.abs(err_db[idx_hi])
        ) / 3.5

        # stay close to specirls warm start
        b_stay = torch.mean((b_param - b0_t).square())
        a_stay = torch.mean((a_full - a0_t).square())
        stay_loss = stay_weight * (b_stay + a_stay)

        # tiny regularization just to suppress absurd coefficient blowup
        reg = 1e-8 * torch.mean(b_param.square()) + 1e-8 * torch.mean(a_full.square())

        loss = db_loss + slope_loss + relmag_loss + softworst_loss + anchor_loss + stay_loss + reg

        stats = {
            "db_loss": float(db_loss.detach().cpu()),
            "slope_loss": float(slope_loss.detach().cpu()),
            "relmag_loss": float(relmag_loss.detach().cpu()),
            "softworst_loss": float(softworst_loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "stay_loss": float(stay_loss.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
        return loss, stats

    def numpy_eval_from_current():
        b_np = b_param.detach().cpu().numpy().astype(np.float64)
        a_np = np.concatenate([[1.0], a_free.detach().cpu().numpy().astype(np.float64)])
        if abs(a_np[0]) < 1e-12:
            a_np[0] = 1.0
        b_np = b_np / a_np[0]
        a_np = a_np / a_np[0]

        _, h = signal.freqz(b_np, a_np, worN=w_grid)
        mag = np.abs(h)
        mag_db = 20.0 * np.log10(mag + 1e-12)
        err_db = mag_db - mag_target_db

        # internal selection score mirrors training objective roughly
        w_vis_np = (vis_floor + (1.0 - vis_floor) /
                    (1.0 + np.exp(-(mag_target_db - vis_db_focus) / max(vis_db_soft, 1e-6))))
        w_vis_np = w_vis_np / np.mean(w_vis_np)

        db_term = np.mean(w_vis_np * (err_db ** 2))
        if slope_weight > 0:
            derr = np.gradient(err_db, np.log(np.maximum(w_grid, 1e-8)))
            slope_term = slope_weight * np.mean(w_vis_np * (derr ** 2))
        else:
            slope_term = 0.0
        relmag_term = relmag_weight * np.mean(
            w_vis_np * (((mag - mag_target) / (mag_target + 0.05)) ** 2)
        )
        anchor_term = 0.10 * (
            abs(err_db[idx_lo]) + 1.5 * abs(err_db[idx_peak]) + abs(err_db[idx_hi])
        ) / 3.5

        score = db_term + slope_term + relmag_term + anchor_term

        return b_np, a_np, score, err_db

    def baseline_score():
        _, h0 = signal.freqz(b0, a0, worN=w_grid)
        mag0 = np.abs(h0)
        mag0_db = 20.0 * np.log10(mag0 + 1e-12)
        err0_db = mag0_db - mag_target_db

        w_vis_np = (vis_floor + (1.0 - vis_floor) /
                    (1.0 + np.exp(-(mag_target_db - vis_db_focus) / max(vis_db_soft, 1e-6))))
        w_vis_np = w_vis_np / np.mean(w_vis_np)

        db_term = np.mean(w_vis_np * (err0_db ** 2))
        if slope_weight > 0:
            derr = np.gradient(err0_db, np.log(np.maximum(w_grid, 1e-8)))
            slope_term = slope_weight * np.mean(w_vis_np * (derr ** 2))
        else:
            slope_term = 0.0
        relmag_term = relmag_weight * np.mean(
            w_vis_np * (((mag0 - mag_target) / (mag_target + 0.05)) ** 2)
        )
        anchor_term = 0.10 * (
            abs(err0_db[idx_lo]) + 1.5 * abs(err0_db[idx_peak]) + abs(err0_db[idx_hi])
        ) / 3.5

        return db_term + slope_term + relmag_term + anchor_term

    # ------------------------------------------------------------
    # best-so-far tracking
    # ------------------------------------------------------------
    base_score = baseline_score()
    best_score = base_score
    best_b = b0.copy()
    best_a = a0.copy()

    def maybe_update_best():
        nonlocal best_score, best_b, best_a
        b_np, a_np, score, _ = numpy_eval_from_current()
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_b = b_np.copy()
            best_a = a_np.copy()

    # ------------------------------------------------------------
    # phase split
    # ------------------------------------------------------------
    adam_steps = max(12, min(40, num_steps // 6))
    lbfgs1_steps = max(30, num_steps // 2)
    lbfgs2_steps = max(20, num_steps - adam_steps - lbfgs1_steps)

    # ------------------------------------------------------------
    # Phase 1: short Adam warmup
    # ------------------------------------------------------------
    opt_adam = torch.optim.Adam(params, lr=lr)

    for _ in range(adam_steps):
        opt_adam.zero_grad()
        loss, _ = internal_loss(
            stage="adam",
            stay_weight=1e-3,
            slope_w=0.5 * slope_weight,
            relmag_w=0.6 * relmag_weight,
            softworst_w=0.00002,
            anchor_w=0.00005,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=20.0)
        opt_adam.step()
        maybe_update_best()

    # ------------------------------------------------------------
    # Phase 2: main LBFGS
    # ------------------------------------------------------------
    opt_lbfgs1 = torch.optim.LBFGS(
        params,
        lr=0.8,
        max_iter=lbfgs1_steps,
        max_eval=max(2 * lbfgs1_steps, 100),
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure1():
        opt_lbfgs1.zero_grad()
        loss, _ = internal_loss(
            stage="lbfgs1",
            stay_weight=5e-4,
            slope_w=slope_weight,
            relmag_w=relmag_weight,
            softworst_w=0.05,
            anchor_w=0.08,
        )
        loss.backward()
        return loss

    opt_lbfgs1.step(closure1)
    maybe_update_best()

    # ------------------------------------------------------------
    # Phase 3: tighter LBFGS, weaker stay-close, more shape-aware
    # ------------------------------------------------------------
    opt_lbfgs2 = torch.optim.LBFGS(
        params,
        lr=0.35,
        max_iter=lbfgs2_steps,
        max_eval=max(2 * lbfgs2_steps, 60),
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
        history_size=30,
        line_search_fn="strong_wolfe",
    )

    def closure2():
        opt_lbfgs2.zero_grad()
        loss, _ = internal_loss(
            stage="lbfgs2",
            stay_weight=1e-4,
            slope_w=1.2 * slope_weight,
            relmag_w=0.8 * relmag_weight,
            softworst_w=0.08,
            anchor_w=0.12,
        )
        loss.backward()
        return loss

    opt_lbfgs2.step(closure2)
    maybe_update_best()

    # ------------------------------------------------------------
    # accept / reject
    # ------------------------------------------------------------
    # If refinement is not better, return baseline.
    #if not np.isfinite(best_score) or best_score >= base_score:
    #    return b0, a0

    # final gain correction on accepted refined result
    _, h_best = signal.freqz(best_b, best_a, worN=w_grid)
    gain_correction = np.mean(mag_target) / (np.mean(np.abs(h_best)) + 1e-12)
    best_b = best_b * gain_correction

    return best_b, best_a

def evaluate_filter(b, a, R_target, w_grid):
    target_mag = np.sqrt(np.maximum(R_target, 0.0))
    _, h = signal.freqz(b, a, worN=w_grid)
    mag = np.abs(h)
    err_db = 20 * np.log10(mag + 1e-8) - 20 * np.log10(target_mag + 1e-8)
    return {
        "mse_linear": float(np.mean((mag - target_mag) ** 2)),
        "rmse_db": float(np.sqrt(np.mean(err_db ** 2))),
        "max_db_abs": float(np.max(np.abs(err_db))),
        "max_pole_radius": float(np.max(np.abs(np.roots(a)))),
    }

if __name__ == "__main__":
    fs = 48000
    num_points = 32768  # 对数分布下，8192 个点已经非常足够且精确了
    
    # 因为对数分布不能包含 0，我们从一个极低的频率(例如 10Hz)开始
    f_min = 70.0 
    w_min = 2 * np.pi * f_min / fs
    w_grid = np.geomspace(w_min, np.pi, num_points)
        
    freqs = w_grid * fs / (2 * np.pi)

    # 定义一个模拟频响(或模拟幅度响应)
    fc = 500             # 截止频率 (Hz)
    wc = 2 * np.pi * fc    # 必须转换为模拟角频率 (rad/s)
    Q = 1
    
    s = 1j * 2 * np.pi * freqs 
    stages = 3 #必须接近1阶(1个二阶系统)，拟合在高低频才有解。如果大于1阶，改之后的iir的拟合阶数也不能很好地拟合低频
    def Hs_peaking(s, A_target = 2):
        w = np.abs(s.imag)
        B = wc / Q
        x = np.abs((w**2 - wc**2) / (B * w + 1e-200))
        W = 1 / (1 + x**(2*stages))
        mag = 1 + (A_target - 1) * W
        return mag
    def Hs_lowpass(s):
        w = np.abs(s.imag)
        if Q > 1 / np.sqrt(2):
            peak_factor = 1 - 1 / (2 * Q**2)
        else:
            peak_factor = 1.0 # 如果 Q 太小没有峰值，则不作频移补偿
        wc_comp = wc * (peak_factor)**(0.5 - 1 / (2 * stages))
        y = w / wc_comp
        denominator = (1 - y**(2 * stages)) + 1j * (y**stages / Q)
        h = 1 / denominator
        return h
    
    Hs = Hs_peaking
    mag_target = np.abs(Hs(s))
    
    # 转换为算法需要的能量响应 |H(w)|^2
    R_target = mag_target**2
    # 拟合目标响应
    target_order = 4
    b_dig1, a_dig1 =  design_matched_iir_from_response(R_target, w_grid, order=target_order)
    b_dig2, a_dig2 =  design_matched_iir_from_response_nn2(R_target, w_grid, order=target_order)
    
    
    # ==========================================
    # 2. 观测阶段：使用高密度网格还原真实的连续曲线
    # ==========================================
    num_plot_points = 65536  # 无论拟合用了多少点，画图都用 2048 点保证平滑
    w_plot = np.linspace(0, np.pi, num_plot_points)
    freqs_plot = w_plot * fs / (2 * np.pi)

    # 用高密度网格重新计算模拟原型的真实响应 (作为完美的黑虚线基准)
    s_plot = 1j * 2 * np.pi * freqs_plot
    Hs_plot = Hs(s_plot)
    mag_target_plot = np.abs(Hs_plot)

    # 用高密度网格评估我们算出的数字滤波器系数的实际响应 (蓝实线)
    _, h_dig_plot1 = signal.freqz(b_dig1, a_dig1, worN=w_plot)
    _, h_dig_plot2 = signal.freqz(b_dig2, a_dig2, worN=w_plot)

    # ==========================================
    # 3. 绘图对比
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_plot, 20 * np.log10(mag_target_plot), 'k--', linewidth=2, label='Analog Prototype Response')
    plt.plot(freqs_plot, 20 * np.log10(np.abs(h_dig_plot1)), 'b-', linewidth=2, alpha=0.8, label=f'Digital IIR Fit specirls({target_order}th order)')
    plt.plot(freqs_plot, 20 * np.log10(np.abs(h_dig_plot2)), 'r-', linewidth=2, alpha=0.8, label=f'Digital IIR Fit nn ({target_order}th order)')

    
    plt.title('Arbitrary Magnitude Target Fitting (Decoupled Evaluation)', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    
    plt.xlim(100, 24000)
    plt.ylim(-30, 30)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    print(f"=== 生成的 {target_order} 阶数字滤波器系数 ===")
    print("b_specirls =", np.array2string(b_dig1, precision=6))
    print("a_specirls =", np.array2string(a_dig1, precision=6))
    print("b_nn =", np.array2string(b_dig2, precision=6))
    print("a_nn =", np.array2string(a_dig2, precision=6))