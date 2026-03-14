import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ============================================================
# Source directory
# ============================================================
SRC_DIR = r"D:\Projects\Py\matchedfilterdesign\iirCoeffsLuts"
os.chdir(SRC_DIR)

BEST_FILE = os.path.join(SRC_DIR, "best.pt")
LUT_FILE = os.path.join(SRC_DIR, "iirluts2b.txt")


# ============================================================
# Model
# ============================================================
class TinyPZIIRNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Normalizer
# ============================================================
class ParamNormalizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_fc_min = math.log(cfg["fc_min"])
        self.log_fc_max = math.log(cfg["fc_max"])
        self.log_q_min = math.log(cfg["q_min"])
        self.log_q_max = math.log(cfg["q_max"])

    def encode_np(self, x):
        fc, q, g, s = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        x0 = (np.log(fc) - self.log_fc_min) / (self.log_fc_max - self.log_fc_min) * 2.0 - 1.0
        x1 = (np.log(q) - self.log_q_min) / (self.log_q_max - self.log_q_min) * 2.0 - 1.0
        x2 = (g - self.cfg["gain_min"]) / (self.cfg["gain_max"] - self.cfg["gain_min"]) * 2.0 - 1.0
        x3 = (s - self.cfg["stages_min"]) / (self.cfg["stages_max"] - self.cfg["stages_min"]) * 2.0 - 1.0
        return np.stack([x0, x1, x2, x3], axis=-1).astype(np.float32)


# ============================================================
# Helpers
# ============================================================
def make_log_freq_grid(fs, f_min_resp, num_freq):
    w_min = 2.0 * np.pi * f_min_resp / fs
    w_grid = np.logspace(np.log10(w_min), np.log10(np.pi), num_freq)
    freqs = w_grid * fs / (2.0 * np.pi)
    return w_grid, freqs


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def prototype_mag_response_numpy(fc, Q, gain_db, stages, freqs_hz):
    wc = 2.0 * np.pi * fc
    omega = 2.0 * np.pi * freqs_hz
    A_target = 10.0 ** (gain_db / 20.0)

    num = np.abs((omega ** 2 - wc ** 2) / (wc / (Q + 1e-12) * omega + 1e-200))
    denom = 1.0 + np.power(num + 1e-30, 2.0 * stages)
    mag = 1.0 + (A_target - 1.0) * (1.0 / denom)
    return mag


def iir_mag_response_numpy(b, a, w_grid):
    k = np.arange(5, dtype=np.float64)
    ejwk = np.exp(-1j * w_grid[:, None] * k[None, :])
    num = np.sum(b[None, :] * ejwk, axis=1)
    den = np.sum(a[None, :] * ejwk, axis=1)
    H = num / (den + 1e-12)
    return np.abs(H)


# ============================================================
# LUT parsing
# ============================================================
def parse_lut_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 19:
                continue

            vals = list(map(float, parts))
            dims = np.array(vals[0:4], dtype=np.int32)
            proto = np.array(vals[4:8], dtype=np.float64)   # [fc, Q, gain_db, stages]
            err = float(vals[8])
            b = np.array(vals[9:14], dtype=np.float64)
            a = np.array(vals[14:19], dtype=np.float64)

            rows.append({
                "dims": dims,
                "proto": proto,
                "err": err,
                "b": b,
                "a": a,
            })

    if not rows:
        raise RuntimeError(f"No valid LUT rows found in {path}")
    return rows


# ============================================================
# NN coeff mapping
# ============================================================
def poly2_mul_np(a1, a2, b1, b2):
    c0 = 1.0
    c1 = a1 + b1
    c2 = a2 + a1 * b1 + b2
    c3 = a1 * b2 + a2 * b1
    c4 = a2 * b2
    return np.array([c0, c1, c2, c3, c4], dtype=np.float64)


def biquad_from_r_theta_np(r, theta):
    c1 = -2.0 * r * np.cos(theta)
    c2 = r * r
    return c1, c2


def coeffs_from_raw_numpy(raw, cfg):
    zr1_raw, zt1_raw, zr2_raw, zt2_raw, pr1_raw, pt1_raw, pr2_raw, pt2_raw, g_raw = raw

    rz1 = cfg["zero_r_max"] * (1.0 / (1.0 + np.exp(-zr1_raw)))
    rz2 = cfg["zero_r_max"] * (1.0 / (1.0 + np.exp(-zr2_raw)))

    pole_r_max = cfg["pole_r_max"]
    rp1 = 1e-4 + (pole_r_max - 1e-4) * (1.0 / (1.0 + np.exp(-pr1_raw)))
    rp2 = 1e-4 + (pole_r_max - 1e-4) * (1.0 / (1.0 + np.exp(-pr2_raw)))

    thz1 = np.pi * (1.0 / (1.0 + np.exp(-zt1_raw)))
    thz2 = np.pi * (1.0 / (1.0 + np.exp(-zt2_raw)))
    thp1 = np.pi * (1.0 / (1.0 + np.exp(-pt1_raw)))
    thp2 = np.pi * (1.0 / (1.0 + np.exp(-pt2_raw)))

    bz1, bz2 = biquad_from_r_theta_np(rz1, thz1)
    bz3, bz4 = biquad_from_r_theta_np(rz2, thz2)
    az1, az2 = biquad_from_r_theta_np(rp1, thp1)
    az3, az4 = biquad_from_r_theta_np(rp2, thp2)

    b = poly2_mul_np(bz1, bz2, bz3, bz4)
    a = poly2_mul_np(az1, az2, az3, az4)

    g = np.exp(cfg["gain_scale_exp"] * g_raw)
    b = b * g

    aux = {
        "rz1": rz1, "rz2": rz2,
        "rp1": rp1, "rp2": rp2,
        "thz1": thz1, "thz2": thz2,
        "thp1": thp1, "thp2": thp2,
        "g": g,
    }
    return b, a, aux


# ============================================================
# Checkpoint loading
# ============================================================
def load_checkpoint_model(ckpt_path, device="cpu"):
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    hidden = int(cfg.get("hidden", 32))

    model = TinyPZIIRNet(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def predict_raw(model, normalizer, fc, Q, gain_db, stages, device="cpu"):
    x = np.array([[fc, Q, gain_db, stages]], dtype=np.float32)
    x_n = normalizer.encode_np(x)
    xt = torch.tensor(x_n, dtype=torch.float32, device=device)

    with torch.no_grad():
        raw = model(xt)[0].cpu().numpy()
    return raw


# ============================================================
# Root / pole-zero tools
# ============================================================
def select_upper_conjugate_members(roots_arr):
    """
    从 roots 里选出上半平面/实轴代表根，期望得到 2 个。
    对 4 阶实系数多项式，复根成对出现。
    """
    roots_arr = np.asarray(roots_arr, dtype=np.complex128)

    selected = []

    # 优先选 imag > 0
    for r in roots_arr:
        if np.imag(r) > 1e-8:
            selected.append(r)

    # 如果数量不够，再补实根
    if len(selected) < 2:
        real_roots = [r for r in roots_arr if abs(np.imag(r)) <= 1e-8]
        # 按角度或半径随便补，这里按半径
        real_roots = sorted(real_roots, key=lambda x: abs(x))
        for r in real_roots:
            selected.append(np.real(r))
            if len(selected) >= 2:
                break

    if len(selected) != 2:
        # 退化情况，选前两个最接近上半平面的
        roots_sorted = sorted(roots_arr, key=lambda x: (-np.imag(x), abs(x)))
        selected = roots_sorted[:2]

    selected = np.array(selected, dtype=np.complex128)
    return selected


def roots_to_r_theta_pair(poly_coeff_desc):
    """
    poly coeff in descending powers, e.g. np.roots usable format.
    return:
        r[2], theta[2]
    """
    roots_arr = np.roots(poly_coeff_desc)
    reps = select_upper_conjugate_members(roots_arr)

    r = np.abs(reps)
    theta = np.abs(np.angle(reps))

    # sort by theta
    idx = np.argsort(theta)
    r = r[idx]
    theta = theta[idx]
    return r, theta


def r_theta_pair_to_poly_desc(r, theta):
    """
    从两对共轭根重建 4 阶降幂多项式
    roots in z-plane:
        r*exp(+j theta), r*exp(-j theta)
    """
    roots_list = []
    for rk, thk in zip(r, theta):
        roots_list.append(rk * np.exp(1j * thk))
        roots_list.append(rk * np.exp(-1j * thk))

    poly_desc = np.real_if_close(np.poly(np.array(roots_list, dtype=np.complex128)))
    poly_desc = np.real(poly_desc)
    return poly_desc


def short_angle_interp(theta0, theta1, t):
    """
    在 [0, pi] 里做简单短弧插值。
    对本场景通常够用。
    """
    d = theta1 - theta0
    return theta0 + t * d


def interp_pz_two_filters(row_lo, row_hi, fc_target):
    """
    用两个 LUT 基底做 pole-zero 参数插值。
    这里插的是：
      - zeros: r, theta
      - poles: r, theta
      - gain-like front scale using b0 ratio
    """
    fc_lo = row_lo["proto"][0]
    fc_hi = row_hi["proto"][0]

    if abs(fc_hi - fc_lo) < 1e-12:
        t = 0.0
    else:
        t = (fc_target - fc_lo) / (fc_hi - fc_lo)
    t = float(np.clip(t, 0.0, 1.0))

    # 分子/分母转成 z 平面多项式：当前 b,a 是 q=z^-1 升幂形式
    # roots 需要降幂格式，所以翻转
    b_lo_desc = row_lo["b"][::-1]
    a_lo_desc = row_lo["a"][::-1]
    b_hi_desc = row_hi["b"][::-1]
    a_hi_desc = row_hi["a"][::-1]

    rz_lo, thz_lo = roots_to_r_theta_pair(b_lo_desc)
    rp_lo, thp_lo = roots_to_r_theta_pair(a_lo_desc)

    rz_hi, thz_hi = roots_to_r_theta_pair(b_hi_desc)
    rp_hi, thp_hi = roots_to_r_theta_pair(a_hi_desc)

    rz = (1.0 - t) * rz_lo + t * rz_hi
    rp = (1.0 - t) * rp_lo + t * rp_hi
    thz = short_angle_interp(thz_lo, thz_hi, t)
    thp = short_angle_interp(thp_lo, thp_hi, t)

    # 重建降幂多项式
    b_desc = r_theta_pair_to_poly_desc(rz, thz)
    a_desc = r_theta_pair_to_poly_desc(rp, thp)

    # 转回 q=z^-1 升幂
    b_q = b_desc[::-1]
    a_q = a_desc[::-1]

    # 归一化 a0=1
    if abs(a_q[0]) < 1e-12:
        a_q = row_lo["a"].copy()
        b_q = row_lo["b"].copy()
        info = {
            "t": t,
            "fallback": True,
            "fc_lo": fc_lo,
            "fc_hi": fc_hi,
        }
        return b_q, a_q, info

    b_q = b_q / a_q[0]
    a_q = a_q / a_q[0]

    # 用 b0 做一个轻微增益插值锚定
    b0_lo = row_lo["b"][0]
    b0_hi = row_hi["b"][0]
    b0_tgt = (1.0 - t) * b0_lo + t * b0_hi
    if abs(b_q[0]) > 1e-12:
        b_q = b_q * (b0_tgt / b_q[0])

    info = {
        "t": t,
        "fallback": False,
        "fc_lo": fc_lo,
        "fc_hi": fc_hi,
        "rz": rz,
        "rp": rp,
        "thz": thz,
        "thp": thp,
    }
    return np.real(b_q), np.real(a_q), info


# ============================================================
# LUT pair selector
# ============================================================
class LUTPairSelector:
    """
    先按 (Q, gain, stages) 选邻域，再沿 fc 选上下两个基底。
    """
    def __init__(self, rows):
        self.rows = rows
        self.params = np.stack([r["proto"] for r in rows], axis=0)

        self.q_log = np.log(self.params[:, 1])
        self.gain = self.params[:, 2]
        self.stages = self.params[:, 3]

        self.q_log_min = self.q_log.min()
        self.q_log_max = self.q_log.max()
        self.gain_min = self.gain.min()
        self.gain_max = self.gain.max()
        self.stages_min = self.stages.min()
        self.stages_max = self.stages.max()

        self.meta_norm = np.stack([
            (self.q_log - self.q_log_min) / (self.q_log_max - self.q_log_min + 1e-12),
            (self.gain - self.gain_min) / (self.gain_max - self.gain_min + 1e-12),
            (self.stages - self.stages_min) / (self.stages_max - self.stages_min + 1e-12),
        ], axis=1)

    def _meta_query(self, Q, gain_db, stages):
        return np.array([
            (np.log(Q) - self.q_log_min) / (self.q_log_max - self.q_log_min + 1e-12),
            (gain_db - self.gain_min) / (self.gain_max - self.gain_min + 1e-12),
            (stages - self.stages_min) / (self.stages_max - self.stages_min + 1e-12),
        ], dtype=np.float64)

    def select_pair(self, fc, Q, gain_db, stages):
        q = self._meta_query(Q, gain_db, stages)

        d2_meta = np.sum((self.meta_norm - q[None, :]) ** 2, axis=1)

        # 取 Q/gain/stages 最接近的一小批候选
        topk = min(256, len(self.rows))
        cand_idx = np.argsort(d2_meta)[:topk]
        cand_rows = [self.rows[i] for i in cand_idx]
        cand_fc = np.array([r["proto"][0] for r in cand_rows], dtype=np.float64)

        lo_mask = cand_fc <= fc
        hi_mask = cand_fc >= fc

        if np.any(lo_mask):
            lo_idx_local = np.argmax(cand_fc[lo_mask])  # mask 内最大 fc
            lo_candidates = np.where(lo_mask)[0]
            lo_idx = lo_candidates[lo_idx_local]
            row_lo = cand_rows[lo_idx]
        else:
            # 没有下方点，取最小 fc
            lo_idx = int(np.argmin(cand_fc))
            row_lo = cand_rows[lo_idx]

        if np.any(hi_mask):
            hi_idx_local = np.argmin(cand_fc[hi_mask])  # mask 内最小 fc
            hi_candidates = np.where(hi_mask)[0]
            hi_idx = hi_candidates[hi_idx_local]
            row_hi = cand_rows[hi_idx]
        else:
            # 没有上方点，取最大 fc
            hi_idx = int(np.argmax(cand_fc))
            row_hi = cand_rows[hi_idx]

        return row_lo, row_hi


# ============================================================
# GUI
# ============================================================
def main():
    device = "cpu"
    model, cfg = load_checkpoint_model(BEST_FILE, device=device)
    normalizer = ParamNormalizer(cfg)

    fs = float(cfg["fs"])
    f_min_resp = float(cfg["f_min_resp"])
    num_freq = 1024
    w_grid, freqs = make_log_freq_grid(fs, f_min_resp, num_freq)

    lut_rows = parse_lut_file(LUT_FILE)
    pair_selector = LUTPairSelector(lut_rows)

    fc0 = 1000.0
    q0 = 2.0
    g0 = 6.0
    s0 = 2.0

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(left=0.08, bottom=0.34, right=0.98, top=0.95)

    proto_mag0 = prototype_mag_response_numpy(fc0, q0, g0, s0, freqs)

    raw0 = predict_raw(model, normalizer, fc0, q0, g0, s0, device=device)
    b_nn0, a_nn0, aux0 = coeffs_from_raw_numpy(raw0, cfg)
    nn_mag0 = iir_mag_response_numpy(b_nn0, a_nn0, w_grid)

    row_lo0, row_hi0 = pair_selector.select_pair(fc0, q0, g0, s0)
    b_pz0, a_pz0, info0 = interp_pz_two_filters(row_lo0, row_hi0, fc0)
    pz_mag0 = iir_mag_response_numpy(b_pz0, a_pz0, w_grid)

    line_proto, = ax.plot(freqs, db(proto_mag0), linewidth=2.0, label="Analog prototype")
    line_nn, = ax.plot(freqs, db(nn_mag0), linewidth=2.0, label="NN IIR response")
    line_pz, = ax.plot(freqs, db(pz_mag0), linewidth=2.0, linestyle="-.", label="Pole-zero interpolated LUT")

    ax.set_xscale("log")
    ax.set_xlim([20, fs / 2.0])
    ax.set_ylim([-24, 24])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Analog Prototype vs NN vs Pole-Zero Interpolated LUT")
    ax.legend(loc="upper right")

    text_box = ax.text(
        0.43, 0.03, "",
        transform=ax.transAxes,
        fontsize=8.7,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax_fc = plt.axes([0.10, 0.24, 0.80, 0.03])
    ax_q  = plt.axes([0.10, 0.19, 0.80, 0.03])
    ax_g  = plt.axes([0.10, 0.14, 0.80, 0.03])
    ax_s  = plt.axes([0.10, 0.09, 0.80, 0.03])

    s_fc = Slider(ax_fc, "fc (Hz)", cfg["fc_min"], cfg["fc_max"], valinit=fc0)
    s_q  = Slider(ax_q,  "Q", cfg["q_min"], cfg["q_max"], valinit=q0)
    s_g  = Slider(ax_g,  "gain (dB)", cfg["gain_min"], cfg["gain_max"], valinit=g0)
    s_s  = Slider(ax_s,  "stages", cfg["stages_min"], cfg["stages_max"], valinit=s0)

    def update(_):
        fc = float(s_fc.val)
        Q = float(s_q.val)
        gain_db = float(s_g.val)
        stages = float(s_s.val)

        proto_mag = prototype_mag_response_numpy(fc, Q, gain_db, stages, freqs)

        raw = predict_raw(model, normalizer, fc, Q, gain_db, stages, device=device)
        b_nn, a_nn, aux = coeffs_from_raw_numpy(raw, cfg)
        nn_mag = iir_mag_response_numpy(b_nn, a_nn, w_grid)

        row_lo, row_hi = pair_selector.select_pair(fc, Q, gain_db, stages)
        b_pz, a_pz, info = interp_pz_two_filters(row_lo, row_hi, fc)
        pz_mag = iir_mag_response_numpy(b_pz, a_pz, w_grid)

        line_proto.set_ydata(db(proto_mag))
        line_nn.set_ydata(db(nn_mag))
        line_pz.set_ydata(db(pz_mag))

        text_box.set_text(
            f"NN  b = [{b_nn[0]: .5f}, {b_nn[1]: .5f}, {b_nn[2]: .5f}, {b_nn[3]: .5f}, {b_nn[4]: .5f}]\n"
            f"NN  a = [{a_nn[0]: .5f}, {a_nn[1]: .5f}, {a_nn[2]: .5f}, {a_nn[3]: .5f}, {a_nn[4]: .5f}]\n"
            f"PZ pair fc_lo={info['fc_lo']:.4f}, fc_hi={info['fc_hi']:.4f}, t={info['t']:.4f}\n"
            f"row_lo idx={row_lo['dims'].tolist()}  err={row_lo['err']:.6f}\n"
            f"row_hi idx={row_hi['dims'].tolist()}  err={row_hi['err']:.6f}\n"
            f"fallback={info['fallback']}"
        )

        fig.canvas.draw_idle()

    s_fc.on_changed(update)
    s_q.on_changed(update)
    s_g.on_changed(update)
    s_s.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()