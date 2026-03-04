import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def spectral_factorization(coeffs):
    """
    对余弦多项式 P(w) = c0 + c1*cos(w) + c2*cos(2w) + ... 进行数值谱分解。
    返回提取到单位圆内的多项式系数。
    """
    N = len(coeffs) - 1
    # 1. 构造 2N 阶的对称 Z 域多项式
    # 对应的 Z 次幂为 z^N, z^(N-1)... z^0 ... z^(-N)
    poly_z = np.zeros(2 * N + 1)
    poly_z[N] = coeffs[0]  # 中心常数项
    for i in range(1, N + 1):
        poly_z[N - i] = coeffs[i] / 2.0  # z^i
        poly_z[N + i] = coeffs[i] / 2.0  # z^-i

    # 2. 求出所有的 2N 个数值根
    roots = np.roots(poly_z)

    # 3. 反射与提取：按照绝对值大小排序，无脑提取最小的那 N 个根
    # （因为根是成对出现的，最小的一半必定全部 <= 1，即在单位圆内）
    sorted_indices = np.argsort(np.abs(roots))
    selected_roots = roots[sorted_indices[:N]]

    # 4. 从圆内的根重建稳定的多项式 (z - r1)(z - r2)...
    factor_poly = np.poly(selected_roots)

    # 丢弃极小的数值计算虚部，保证系数为实数
    return np.real(factor_poly)


def design_matched_iir_4th_order(b_a, a_a, fs, num_points=4096):
    """
    输入:
        b_a, a_a : 2阶模拟滤波器的分子分母系数 (S域)
        fs       : 采样率
    输出:
        b, a     : 完美拟合模拟频响的 4阶数字 IIR 滤波器系数 (Z域)
    """
    # === 第一步：生成目标频响曲线 ===
    w_grid = np.linspace(0, np.pi, num_points)
    Omega_grid = w_grid * fs  # 真实的模拟角频率

    # 计算模拟原型的绝对能量响应 R_target = |H(j * Omega)|^2
    _, H_a = signal.freqs(b_a, a_a, Omega_grid)
    R_target = np.abs(H_a)**2

    # === 第二步：通过最小二乘法进行有理多项式曲线拟合 ===
    # 我们要解方程: P(w) - R_target(w) * Q(w) = 0
    # 其中 P(w) = c0 + c1*cos(w)...c4*cos(4w), Q(w) = 1 + d1*cos(w)...d4*cos(4w)
    
    A = np.zeros((num_points, 9))
    A[:, 0] = 1.0  # 对应 c0
    for k in range(1, 5):
        A[:, k] = np.cos(k * w_grid)               # 对应 c_k
        A[:, k+4] = -R_target * np.cos(k * w_grid) # 对应 d_k

    b_vec = R_target # 将等式右边的 R_target*1 移过去

    # 使用 NumPy 求解超定线性方程组 A * x = b
    x, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)

    # 分离出拟合得到的 c 和 d 系数
    c = x[0:5]
    d = np.concatenate(([1.0], x[5:9]))

    # === 第三步：工程安全兜底（强制非负） ===
    # 极其重要：如果数值拟合在极高频出现了微小的负数，谱分解必定失败！
    P_vals = c[0] + sum(c[k]*np.cos(k*w_grid) for k in range(1, 5))
    Q_vals = d[0] + sum(d[k]*np.cos(k*w_grid) for k in range(1, 5))
    
    if np.min(P_vals) <= 0:
        c[0] += abs(np.min(P_vals)) + 1e-8  # 抬高一点点保证绝对大于0
    if np.min(Q_vals) <= 0:
        d[0] += abs(np.min(Q_vals)) + 1e-8

    # === 第四步：执行谱分解提取稳定解 ===
    b_unscaled = spectral_factorization(c)
    a_final = spectral_factorization(d)

    # === 第五步：增益校准 ===
    # 随便找一个参考点对齐能量，这里选择目标频响能量最大的点
    idx_max = np.argmax(R_target)
    w_ref = w_grid[idx_max]
    z_ref = np.exp(1j * w_ref)
    
    # 评测我们拆出来的数字多项式在这个频率的增益
    mag_num = np.abs(np.polyval(b_unscaled, z_ref))
    mag_den = np.abs(np.polyval(a_final, z_ref))
    mag_digital = mag_num / mag_den
    
    mag_analog = np.sqrt(R_target[idx_max])
    gain_correction = mag_analog / (mag_digital + 1e-12)

    b_final = b_unscaled * gain_correction

    return b_final, a_final

# ================= 测试与绘图对比 =================
if __name__ == "__main__":
    fs = 48000
    # 我们故意设计一个 15kHz 的 2阶低通滤波器
    # 这种高频下，传统的双线性变换会产生极其严重的频率扭曲
    f0 = 22000  
    w0 = 2 * np.pi * f0
    Q = 10.0
    
    # 2阶模拟低通参数 (w0^2 / (s^2 + s*(w0/Q) + w0^2))
    b_a = [w0**2]
    a_a = [1.0, w0/Q, w0**2]

    # 1. 使用我们的自研 4阶设计器
    b_4th, a_4th = design_matched_iir_4th_order(b_a, a_a, fs)
    
    # 2. 作为反面教材：使用标准 SciPy 的双线性变换 (2阶)
    # 注意：这里我们故意不做预映射，以展示纯 BLT 的扭曲
    b_blt, a_blt = signal.bilinear(b_a, a_a, fs=fs)

    # 评估三种滤波器的频响
    freqs = np.linspace(10, 24000, 1000)
    w_digital = 2 * np.pi * freqs / fs
    
    _, h_analog = signal.freqs(b_a, a_a, 2 * np.pi * freqs)
    _, h_4th = signal.freqz(b_4th, a_4th, worN=w_digital)
    _, h_blt = signal.freqz(b_blt, a_blt, worN=w_digital)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(np.abs(h_analog)), 'k--', linewidth=2, label='Target Analog Prototype (2nd order)')
    plt.plot(freqs, 20 * np.log10(np.abs(h_blt)), 'r-', alpha=0.7, label='Standard BLT (2nd order) - Note the warping!')
    plt.plot(freqs, 20 * np.log10(np.abs(h_4th)), 'b-', linewidth=2, label='Our 4th-order Spectral Factorization Fit')
    
    plt.axvline(f0, color='gray', linestyle=':', label=f'Cutoff Frequency ({f0} Hz)')
    plt.title('Magnitude Response: Analog Prototype vs. Digital Filters', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.xlim(200, 24000)
    plt.ylim(-40, 40)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print("=== 生成的 4 阶数字滤波器系数 ===")
    print("b =", np.array2string(b_4th, precision=6))
    print("a =", np.array2string(a_4th, precision=6))