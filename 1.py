import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def spectral_factorization(coeffs):
    """
    对余弦多项式 P(w) = c0 + c1*cos(w) + c2*cos(2w) + ... 进行数值谱分解。
    """
    N = len(coeffs) - 1
    poly_z = np.zeros(2 * N + 1)
    poly_z[N] = coeffs[0]  
    for i in range(1, N + 1):
        poly_z[N - i] = coeffs[i] / 2.0  
        poly_z[N + i] = coeffs[i] / 2.0  

    roots = np.roots(poly_z)
    sorted_indices = np.argsort(np.abs(roots))
    selected_roots = roots[sorted_indices[:N]]
    factor_poly = np.poly(selected_roots)

    return np.real(factor_poly)

def design_matched_iir_from_response(R_target, w_grid, order=4):
    """
    输入:
        R_target : 目标频响的绝对能量响应 |H(w)|^2，长度需与 w_grid 一致
        w_grid   : 数字角频率网格 [0, pi] (单位: rad/sample)
        order    : 期望拟合的数字 IIR 滤波器阶数
    输出:
        b, a     : 完美拟合目标曲线的数字 IIR 滤波器系数 (Z域)
    """
    N = order
    num_points = len(w_grid)

    if len(R_target) != num_points:
        raise ValueError("R_target 和 w_grid 的长度必须一致！")

    # === 迭代重加权最小二乘法进行拟合 ===
    num_iters = 3  
    W = np.ones(num_points)  
    
    for iteration in range(num_iters):
        A = np.zeros((num_points, 2 * N + 1))
        A[:, 0] = W * 1.0
        
        for k in range(1, N + 1):
            A[:, k] = W * np.cos(k * w_grid)
            A[:, k + N] = -W * R_target * np.cos(k * w_grid)
            
        b_vec = W * R_target 
        
        x, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        
        d_temp = np.concatenate(([1.0], x[N + 1 : 2 * N + 1]))
        Q_vals = d_temp[0] + sum(d_temp[k] * np.cos(k * w_grid) for k in range(1, N + 1))
        W = 1.0 / (np.abs(Q_vals) + 1e-8)

    c = x[0 : N + 1]
    d = np.concatenate(([1.0], x[N + 1 : 2 * N + 1]))

    # === 工程安全兜底（强制非负） ===
    P_vals = c[0] + sum(c[k] * np.cos(k * w_grid) for k in range(1, N + 1))
    Q_vals = d[0] + sum(d[k] * np.cos(k * w_grid) for k in range(1, N + 1))
    
    if np.min(P_vals) <= 0:
        c[0] += abs(np.min(P_vals)) + 1e-8  
    if np.min(Q_vals) <= 0:
        d[0] += abs(np.min(Q_vals)) + 1e-8

    # === 执行谱分解提取稳定解 ===
    b_unscaled = spectral_factorization(c)
    a_final = spectral_factorization(d)

     # === 第五步：增益校准（平均线性增益补偿） ===
    # 1. 计算目标曲线在整个频率网格上的平均线性幅度
    # （注意：传入的 R_target 是能量响应，需开方还原为线性幅度）
    mean_mag_target = np.mean(np.sqrt(R_target))
    # 2. 计算当前未校准 IIR 滤波器在相同 w_grid 上的真实线性幅度
    # 使用 freqz 批量计算整个数组，比写 for 循环逐个 polyval 更高效、更数值安全
    _, h_unscaled = signal.freqz(b_unscaled, a_final, worN=w_grid)
    mean_mag_digital = np.mean(np.abs(h_unscaled))
    # 3. 计算补偿比例并缩放分子系数
    gain_correction = mean_mag_target / (mean_mag_digital + 1e-12)
    b_final = b_unscaled * gain_correction

    return b_final, a_final

# ================= 测试与绘图：拟合任意自定义曲线 =================
if __name__ == "__main__":
    fs = 48000
    num_points = 512
    w_grid = np.linspace(0, np.pi, num_points)
    freqs = w_grid * fs / (2 * np.pi)

    # 定义一个模拟二阶低通的频响
    fc = 23000             # 截止频率 (Hz)
    wc = 2 * np.pi * fc    # 必须转换为模拟角频率 (rad/s)
    Q = 20
    
    s = 1j * 2 * np.pi * freqs 
    Hs = wc**2 / (s**2 + s * wc / Q + wc**2)
    mag_target = np.abs(Hs)
    
    # 转换为算法需要的能量响应 |H(w)|^2
    R_target = mag_target**2
    # 拟合目标响应
    target_order = 4
    b_dig, a_dig = design_matched_iir_from_response(R_target, w_grid, order=target_order)
    
    # 评估我们生成的数字滤波器的实际频响
    _, h_dig = signal.freqz(b_dig, a_dig, worN=w_grid)

    # 绘图对比
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(mag_target), 'k--', linewidth=2, label='Custom Target Arbitrary Response')
    plt.plot(freqs, 20 * np.log10(np.abs(h_dig)), 'b-', linewidth=2, alpha=0.8, label=f'Our IIR Fit ({target_order}th order)')
    
    plt.title('Arbitrary Magnitude Target Fitting with Direct Least-Squares', fontsize=14)
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
    print("b =", np.array2string(b_dig, precision=6))
    print("a =", np.array2string(a_dig, precision=6))