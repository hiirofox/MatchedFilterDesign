function [b_final, a_final] = design_matched_iir_spectral(R_target, w_grid, order)
    % 输入:
    %   R_target : 目标能量响应 |H(w)|^2
    %   w_grid   : 数字角频率网格 [0, pi]
    %   order    : IIR 阶数 (建议 4)
    
    N = order;
    num_points = length(w_grid);
    
    % === 迭代重加权最小二乘拟合 (IRLS) ===
    num_iters = 6;
    W = ones(num_points, 1);
    
    for iter = 1:num_iters
        % 构造基函数矩阵 A
        % 目标方程: |H|^2 * Q(cos) = P(cos)
        % 其中 P 和 Q 是余弦多项式
        A = zeros(num_points, 2*N + 1);
        A(:, 1) = W * 1.0; % P0
        
        for k = 1:N
            A(:, k+1) = W .* cos(k * w_grid);         % P_k 基
            A(:, k+N+1) = -W .* R_target .* cos(k * w_grid); % Q_k 基
        end
        
        b_vec = W .* R_target;
        
        % 使用 MATLAB 强大的左除算子 (\) 求解线性方程组
        % 它内部会自动进行 QR 或 SVD 分解来应对病态矩阵
        x = A \ b_vec;
        
        % 更新权重以提高拟合精度
        d_temp = [1; x(N+2 : 2*N+1)];
        Q_vals = d_temp(1);
        for k = 1:N
            Q_vals = Q_vals + d_temp(k+1) * cos(k * w_grid);
        end
        W = 1.0 ./ (abs(Q_vals) + 1e-6);
    end
    
    c = x(1 : N+1);        % 分子余弦多项式系数
    d = [1; x(N+2 : end)]; % 分母余弦多项式系数
    
    % === 执行谱分解提取稳定解 ===
    b_unscaled = spectral_factorization(c);
    a_final = spectral_factorization(d);
    
    % === 增益校准 (平均线性增益补偿) ===
    [h_unscaled, ~] = freqz(b_unscaled, a_final, w_grid);
    mean_mag_target = mean(sqrt(R_target));
    mean_mag_digital = mean(abs(h_unscaled));
    
    gain_correction = mean_mag_target / (mean_mag_digital + 1e-12);
    b_final = b_unscaled * gain_correction;
end

function factor_poly = spectral_factorization(coeffs)
    % 对余弦多项式进行谱分解以提取 Z 域稳定根
    N = length(coeffs) - 1;
    poly_z = zeros(2*N + 1, 1);
    
    % 将 cos(kw) 转换为 (z^k + z^-k)/2
    poly_z(N + 1) = coeffs(1); % 常数项
    for k = 1:N
        poly_z(N + 1 - k) = coeffs(k+1) / 2.0;
        poly_z(N + 1 + k) = coeffs(k+1) / 2.0;
    end
    
    % 求根
    r = roots(poly_z);
    
    % 关键：选择单位圆内的根（最小相位性质，保证稳定性）
    % 过滤掉幅度接近 0 的数值噪声
    r = r(abs(r) < 1.0 | abs(abs(r) - 1.0) < 1e-10);
    
    % 按照与单位圆的距离排序，取最里面的 N 个根
    [~, idx] = sort(abs(r));
    selected_roots = r(idx(1:N));
    
    % 从根还原多项式系数
    factor_poly = real(poly(selected_roots));
end



%% 测试主脚本
fs = 48000;
num_points = 8192;
f_min = 70;
% 生成几何分布（对数分布）的频率网格
w_grid = logspace(log10(2 * pi * f_min / fs), log10(pi), num_points)';
freqs = w_grid * fs / (2 * pi);

% 定义目标模拟原型
fc = 2100; 
wc = 2 * pi * fc;
Q = 10;
stages = 2;

% 计算 Hs_lowpass
% --- 修正部分 ---
s = 1j * 2 * pi * freqs; 

% 注意：MATLAB 中取虚部要用 imag(s)，不能用 s.imag
w_abs = abs(imag(s)); 

if Q > 1/sqrt(2)
    peak_factor = 1 - 1/(2 * Q^2);
else
    peak_factor = 1.0;
end

wc_comp = wc * (peak_factor)^(0.5 - 1 / (2 * stages));

% 使用修正后的 w_abs 进行计算
y = w_abs ./ wc_comp;

% 计算分母（注意使用 .^ 进行元素级幂运算）
denominator = (1 - y.^(2 * stages)) + 1j * (y.^stages / Q);
Hs = 1 ./ denominator;
% ----------------

mag_target = abs(Hs);
R_target = mag_target.^2;

% 执行拟合
[b_dig, a_dig] = design_matched_iir_spectral(R_target, w_grid, 4);

%% 绘图
[h_plot, w_plot] = freqz(b_dig, a_dig, 16384, fs);
figure;
semilogx(freqs, 20*log10(mag_target), 'k--', 'LineWidth', 2); hold on;
semilogx(w_plot, 20*log10(abs(h_plot)), 'b-', 'LineWidth', 1.5);
grid on; xlim([100, 24000]); ylim([-30, 30]);
title('MATLAB Spectral Factorization Fitting');
legend('Analog Prototype', 'Digital IIR (Spectral Fact)');