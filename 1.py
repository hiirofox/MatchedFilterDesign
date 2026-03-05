import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def spectral_factorization(coeffs):
    #对余弦多项式 P(w) = c0 + c1*cos(w) + c2*cos(2w) + ... 进行数值谱分解。
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
    #输入:
    #    R_target : 目标频响的绝对能量响应 |H(w)|^2，长度需与 w_grid 一致
    #    w_grid   : 数字角频率网格 [0, pi] (单位: rad/sample)
    #    order    : 期望拟合的数字 IIR 滤波器阶数
    #输出:
    #    b, a     : 完美拟合目标曲线的数字 IIR 滤波器系数 (Z域)
    
    N = order
    num_points = len(w_grid)

    if len(R_target) != num_points:
        raise ValueError("R_target 和 w_grid 的长度必须一致！")

    # === 迭代重加权最小二乘法进行拟合 ===
    num_iters = 6
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
        W = 1.0 / (np.abs(Q_vals) + 1e-6)
        #target_mag_db = 20 * np.log10(np.sqrt(R_target) + 1e-6)
        #importance_weight = np.abs(target_mag_db)*100 + 10.0 # 幅度变化剧烈的地方权重更大
        #W = W * importance_weight / (w_grid + 0.01)     # 频率越低权重越大
        
        
    c = x[0 : N + 1]
    d = np.concatenate(([1.0], x[N + 1 : 2 * N + 1]))

    # === 工程安全兜底（强制非负） ===
    if (False): # 我发现不仅没用，甚至会影响精度。谱分解已经决定稳定下限了
        P_vals = c[0] + sum(c[k] * np.cos(k * w_grid) for k in range(1, N + 1))
        Q_vals = d[0] + sum(d[k] * np.cos(k * w_grid) for k in range(1, N + 1))
    
        if np.min(P_vals) <= 0:
            c[0] += abs(np.min(P_vals)) + 1e-12
        if np.min(Q_vals) <= 0:
            d[0] += abs(np.min(Q_vals)) + 1e-12
    
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
    num_points = 32768  # 对数分布下，8192 个点已经非常足够且精确了
    
    # 因为对数分布不能包含 0，我们从一个极低的频率(例如 10Hz)开始
    f_min = 70.0 
    w_min = 2 * np.pi * f_min / fs
    w_grid = np.geomspace(w_min, np.pi, num_points)
        
    freqs = w_grid * fs / (2 * np.pi)

    # 定义一个模拟频响(或模拟幅度响应)
    fc = 21000             # 截止频率 (Hz)
    wc = 2 * np.pi * fc    # 必须转换为模拟角频率 (rad/s)
    Q = 10
    
    s = 1j * 2 * np.pi * freqs 
    stages = 1 #必须接近1阶(1个二阶系统)，拟合在高低频才有解。如果大于1阶，改之后的iir的拟合阶数也不能很好地拟合低频
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
    
    Hs = Hs_lowpass
    mag_target = np.abs(Hs(s))
    
    # 转换为算法需要的能量响应 |H(w)|^2
    R_target = mag_target**2
    # 拟合目标响应
    target_order = 4
    b_dig, a_dig =  design_matched_iir_from_response(R_target, w_grid, order=target_order)
    
    
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
    _, h_dig_plot = signal.freqz(b_dig, a_dig, worN=w_plot)

    # ==========================================
    # 3. 绘图对比
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_plot, 20 * np.log10(mag_target_plot), 'k--', linewidth=2, label='Analog Prototype Response')
    plt.plot(freqs_plot, 20 * np.log10(np.abs(h_dig_plot)), 'b-', linewidth=2, alpha=0.8, label=f'Digital IIR Fit ({target_order}th order)')
    
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
    print("b =", np.array2string(b_dig, precision=6))
    print("a =", np.array2string(a_dig, precision=6))