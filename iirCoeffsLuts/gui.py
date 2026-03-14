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


def prototype_mag_response_numpy(fc, Q, gain_db, stages, freqs_hz):
    wc = 2.0 * np.pi * fc
    omega = 2.0 * np.pi * freqs_hz
    A_target = 10 ** (gain_db / 20.0)

    num = np.abs((omega ** 2 - wc ** 2) / (wc / (Q + 1e-12) * omega + 1e-200))
    denom = 1.0 + np.power(num + 1e-30, 2.0 * stages)
    mag = 1.0 + (A_target - 1.0) * (1.0 / denom)
    return mag


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


# ============================================================
# LUT parsing / nearest LUT
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
            proto = np.array(vals[4:8], dtype=np.float64)
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


class LUTNearestSearcher:
    def __init__(self, rows):
        self.rows = rows
        self.params = np.stack([r["proto"] for r in rows], axis=0)

        self.fc_log = np.log(self.params[:, 0])
        self.q_log = np.log(self.params[:, 1])
        self.gain = self.params[:, 2]
        self.stages = self.params[:, 3]

        self.fc_log_min = self.fc_log.min()
        self.fc_log_max = self.fc_log.max()
        self.q_log_min = self.q_log.min()
        self.q_log_max = self.q_log.max()
        self.gain_min = self.gain.min()
        self.gain_max = self.gain.max()
        self.stages_min = self.stages.min()
        self.stages_max = self.stages.max()

        self.params_norm = np.stack([
            (self.fc_log - self.fc_log_min) / (self.fc_log_max - self.fc_log_min + 1e-12),
            (self.q_log - self.q_log_min) / (self.q_log_max - self.q_log_min + 1e-12),
            (self.gain - self.gain_min) / (self.gain_max - self.gain_min + 1e-12),
            (self.stages - self.stages_min) / (self.stages_max - self.stages_min + 1e-12),
        ], axis=1)

    def query(self, fc, Q, gain_db, stages):
        q = np.array([
            (np.log(fc) - self.fc_log_min) / (self.fc_log_max - self.fc_log_min + 1e-12),
            (np.log(Q) - self.q_log_min) / (self.q_log_max - self.q_log_min + 1e-12),
            (gain_db - self.gain_min) / (self.gain_max - self.gain_min + 1e-12),
            (stages - self.stages_min) / (self.stages_max - self.stages_min + 1e-12),
        ], dtype=np.float64)

        d2 = np.sum((self.params_norm - q[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return self.rows[idx], idx, float(np.sqrt(d2[idx]))


# ============================================================
# Pole-zero coefficient mapping
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


def iir_mag_response_numpy(b, a, w_grid):
    k = np.arange(5, dtype=np.float64)
    ejwk = np.exp(-1j * w_grid[:, None] * k[None, :])
    num = np.sum(b[None, :] * ejwk, axis=1)
    den = np.sum(a[None, :] * ejwk, axis=1)
    H = num / (den + 1e-12)
    return np.abs(H)


# ============================================================
# Allpass warp
# ============================================================
def poly_add_asc(a, b):
    la = len(a)
    lb = len(b)
    L = max(la, lb)
    aa = np.pad(a, (0, L - la))
    bb = np.pad(b, (0, L - lb))
    return aa + bb


def poly_power_asc(base, n):
    if n == 0:
        return np.array([1.0], dtype=np.float64)
    out = np.array([1.0], dtype=np.float64)
    for _ in range(n):
        out = np.convolve(out, base)
    return out


def apply_allpass_warp_to_iir_numpy(b_in, a_in, awarp):
    b_in = np.asarray(b_in, dtype=np.float64).ravel()
    a_in = np.asarray(a_in, dtype=np.float64).ravel()

    Mb = len(b_in) - 1
    Ma = len(a_in) - 1
    L = max(Mb, Ma)

    b_pad = np.pad(b_in, (0, L - Mb))
    a_pad = np.pad(a_in, (0, L - Ma))

    # q' = (a - q) / (1 - a q), q=z^-1
    p_num = np.array([awarp, -1.0], dtype=np.float64)
    p_den = np.array([1.0, -awarp], dtype=np.float64)

    num_acc = np.array([0.0], dtype=np.float64)
    den_acc = np.array([0.0], dtype=np.float64)

    for k in range(L + 1):
        term_num = poly_power_asc(p_num, k)
        term_den = poly_power_asc(p_den, L - k)
        term = np.convolve(term_num, term_den)

        num_acc = poly_add_asc(num_acc, b_pad[k] * term)
        den_acc = poly_add_asc(den_acc, a_pad[k] * term)

    if abs(den_acc[0]) < 1e-14:
        return b_in.copy(), a_in.copy()

    b_out = num_acc / den_acc[0]
    a_out = den_acc / den_acc[0]
    return b_out, a_out

def warped_center_frequency(fc_lut, fs, a):
    # 在 q = z^-1 域里做映射：q' = (a - q) / (1 - a q)
    # q = exp(-j w), q' = exp(-j w')
    # 所以 w' = -angle(q')
    w = 2.0 * np.pi * fc_lut / fs
    q = np.exp(-1j * w)
    q2 = (a - q) / (1.0 - a * q)

    w2 = -np.angle(q2)

    # fold to [0, pi]
    if w2 < 0.0:
        w2 += 2.0 * np.pi
    if w2 > np.pi:
        w2 = 2.0 * np.pi - w2

    f2 = w2 * fs / (2.0 * np.pi)
    return f2


def find_best_warp_for_fc(fc_lut, fc_target, fs):
    # 更密一点，低频更稳
    a_grid = np.linspace(-0.995, 0.995, 2001)
    f_map = np.array([warped_center_frequency(fc_lut, fs, a) for a in a_grid])

    idx = np.argmin(np.abs(f_map - fc_target))
    return float(a_grid[idx]), float(f_map[idx])


def apply_gain_extension(b, a, gain_db_lut, gain_db_target):
    A_lut = 10.0 ** (gain_db_lut / 20.0)
    A_tgt = 10.0 ** (gain_db_target / 20.0)

    denom = (A_lut - 1.0)

    # 避免非常接近 0 dB 时数值爆炸
    if abs(denom) < 1e-9:
        scale = 1.0
    else:
        scale = (A_tgt - 1.0) / denom

    # H_new = 1 + scale * (H - 1)
    #       = (scale * B + (1 - scale) * A) / A
    b_new = scale * b + (1.0 - scale) * a
    a_new = a.copy()

    return b_new, a_new, scale


# ============================================================
# Load checkpoint
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
# Main GUI
# ============================================================
def main():
    device = "cpu"
    model, cfg = load_checkpoint_model(BEST_FILE, device=device)
    normalizer = ParamNormalizer(cfg)

    lut_rows = parse_lut_file(LUT_FILE)
    lut_searcher = LUTNearestSearcher(lut_rows)

    fs = cfg["fs"]
    f_min_resp = cfg["f_min_resp"]
    num_freq = 1024
    w_grid, freqs = make_log_freq_grid(fs, f_min_resp, num_freq)

    fc0 = 1000.0
    q0 = 2.0
    g0 = 6.0
    s0 = 2.0

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(left=0.08, bottom=0.34, right=0.98, top=0.95)

    proto_mag0 = prototype_mag_response_numpy(fc0, q0, g0, s0, freqs)

    raw0 = predict_raw(model, normalizer, fc0, q0, g0, s0, device=device)
    b0, a0, aux0 = coeffs_from_raw_numpy(raw0, cfg)
    iir_mag0 = iir_mag_response_numpy(b0, a0, w_grid)

    lut_row0, lut_idx0, lut_dist0 = lut_searcher.query(fc0, q0, g0, s0)
    lut_mag0 = iir_mag_response_numpy(lut_row0["b"], lut_row0["a"], w_grid)

    awarp0, fc_map0 = find_best_warp_for_fc(lut_row0["proto"][0], fc0, fs)
    bw0, aw0 = apply_allpass_warp_to_iir_numpy(lut_row0["b"], lut_row0["a"], awarp0)
    bg0, ag0, gain_scale0 = apply_gain_extension(bw0, aw0, lut_row0["proto"][2], g0)
    warped_mag0 = iir_mag_response_numpy(bg0, ag0, w_grid)

    line_proto, = ax.plot(freqs, db(proto_mag0), linewidth=2.0, label="Analog prototype")
    #line_iir, = ax.plot(freqs, db(iir_mag0), linewidth=2.0, label="NN IIR response")
    #line_lut, = ax.plot(freqs, db(lut_mag0), linewidth=1.8, linestyle="--", label="Nearest LUT response")
    line_warp, = ax.plot(freqs, db(warped_mag0), linewidth=2.0, linestyle="-.", label="Warped LUT response")

    ax.set_xscale("log")
    ax.set_xlim([20, fs / 2])
    ax.set_ylim([-24, 24])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Prototype vs NN vs Nearest LUT vs Warped LUT")
    ax.legend(loc="upper right")

    text_box = ax.text(
        0.46, 0.03, "",
        transform=ax.transAxes,
        fontsize=8.5,
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

        # prototype
        proto_mag = prototype_mag_response_numpy(fc, Q, gain_db, stages, freqs)

        # NN
        raw = predict_raw(model, normalizer, fc, Q, gain_db, stages, device=device)
        b, a, aux = coeffs_from_raw_numpy(raw, cfg)
        iir_mag = iir_mag_response_numpy(b, a, w_grid)

        # nearest LUT
        lut_row, lut_idx, lut_dist = lut_searcher.query(fc, Q, gain_db, stages)
        lut_mag = iir_mag_response_numpy(lut_row["b"], lut_row["a"], w_grid)

        # warped LUT
        fc_lut = lut_row["proto"][0]
        gain_lut = lut_row["proto"][2]

        awarp, fc_map = find_best_warp_for_fc(fc_lut, fc, fs)
        bw, aw = apply_allpass_warp_to_iir_numpy(lut_row["b"], lut_row["a"], awarp)
        bg, ag, gain_scale = apply_gain_extension(bw, aw, gain_lut, gain_db)
        warped_mag = iir_mag_response_numpy(bg, ag, w_grid)

        line_proto.set_ydata(db(proto_mag))
        #line_iir.set_ydata(db(iir_mag))
        #line_lut.set_ydata(db(lut_mag))
        line_warp.set_ydata(db(warped_mag))

        lut_fc, lut_Q, lut_g, lut_s = lut_row["proto"]

        text_box.set_text(
            f"NN  b = [{b[0]: .5f}, {b[1]: .5f}, {b[2]: .5f}, {b[3]: .5f}, {b[4]: .5f}]\n"
            f"NN  a = [{a[0]: .5f}, {a[1]: .5f}, {a[2]: .5f}, {a[3]: .5f}, {a[4]: .5f}]\n"
            f"Nearest LUT idx={lut_row['dims'].tolist()}  dist={lut_dist:.6f}  err={lut_row['err']:.6f}\n"
            f"LUT proto = [fc={lut_fc:.4f}, Q={lut_Q:.4f}, gain={lut_g:.4f}, stages={lut_s:.4f}]\n"
            f"warp a={awarp:.5f}, mapped_fc={fc_map:.4f}, gain_scale={gain_scale:.5f}"
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