
function compare_matched_iir_methods()
    clc; close all;

    %% =========================
    % 0. 测试目标
    % ==========================
    fs = 48000;
    num_points = 8192;

    f_min = 70.0;
    w_min = 2 * pi * f_min / fs;
    w_grid = logspace(log10(w_min), log10(pi), num_points)';   % rad/sample
    freqs = w_grid * fs / (2 * pi);                            % Hz

    order = 6;

    %% =========================
    % 1. 定义模拟目标幅度
    % ==========================
    fc = 230;
    wc = 2 * pi * fc;
    Q = 2;
    stages = 6;

    s = 1j * 2 * pi * freqs;
    
    A_target = 2;
    Hs_peaking = @(s) (1 + (A_target - 1) * ...
        (1 ./ (1 + (abs((imag(s).^2 - wc^2) ./ (wc/Q * imag(s) + 1e-200))).^(2*stages))));

    if Q > 1/sqrt(2)
        peak_factor = 1 - 1 / (2 * Q^2);
    else
        peak_factor = 1.0;
    end

    wc_comp = wc * (peak_factor)^(0.5 - 1 / (2 * stages));

    Hs_lowpass = @(s) (1 ./ ( (1 - (abs(imag(s))/wc_comp).^(2*stages)) + ...
        1j * ((abs(imag(s))/wc_comp).^stages / Q) ));

    % 你可以切换目标
    Hs_obj = Hs_lowpass(s);
    mag_target = abs(Hs_obj);
    R_target = mag_target.^2;

    %% =========================
    % 2. 设计多个版本（加入预扭曲优化 a）
    % ==========================
    methods = {};
    results = struct([]);
    warp_infos = struct( ...
    'method',{}, ...
    'a_initial',{}, ...
    'a_final',{}, ...
    'score_initial',{}, ...
    'score_final',{}, ...
    'score_drop',{}, ...
    'score_drop_pct',{}, ...
    'rmse_db_initial',{}, ...
    'rmse_db_final',{}, ...
    'rmse_db_drop',{}, ...
    'rmse_db_drop_pct',{}, ...
    'mse_initial',{}, ...
    'mse_final',{}, ...
    'mse_drop',{}, ...
    'mse_drop_pct',{});

    % ---- 方法 1: spectral-irls ----
    design_fn = @(Rt, wg, ord, fs_) design_matched_iir_spectral(Rt, wg, ord);
    [b1, a1, aopt1, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'spectral-irls');
    warp_infos(end+1) = winfo;
    methods{end+1} = sprintf('spectral-irls (a=%.4f)', aopt1);
    tmp = evaluate_method(methods{end}, b1, a1, R_target, w_grid, fs);
    if isempty(results), results = tmp; else, results(end+1) = tmp; end

    % ---- 方法 2: 最小相位 + invfreqz ----
    design_fn = @(Rt, wg, ord, fs_) design_matched_iir_invfreqz(Rt, wg, ord);
    [b2, a2, aopt2, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'minphase-invfreqz');
    warp_infos(end+1) = winfo;
    methods{end+1} = sprintf('minphase-invfreqz (a=%.4f)', aopt2);
    tmp = evaluate_method(methods{end}, b2, a2, R_target, w_grid, fs);
    if isempty(results), results = tmp; else, results(end+1) = tmp; end

    % ---- 方法 2b: 最小相位恢复 + prony ----
    try
        design_fn = @(Rt, wg, ord, fs_) design_matched_iir_prony_minphase(Rt, wg, ord);
        [b2b, a2b, aopt2b, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'minphase-prony');
        warp_infos(end+1) = winfo;
        methods{end+1} = sprintf('minphase-prony (a=%.4f)', aopt2b);
        results(end+1) = evaluate_method(methods{end}, b2b, a2b, R_target, w_grid, fs);
    catch ME
        warning('minphase-prony failed: %s', ME.message);
    end
    
    % ---- 方法 2c: 最小相位恢复 + stmcb ----
    try
        design_fn = @(Rt, wg, ord, fs_) design_matched_iir_stmcb_minphase(Rt, wg, ord);
        [b2c, a2c, aopt2c, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'minphase-stmcb');
        warp_infos(end+1) = winfo;
        methods{end+1} = sprintf('minphase-stmcb (a=%.4f)', aopt2c);
        results(end+1) = evaluate_method(methods{end}, b2c, a2c, R_target, w_grid, fs);
    catch ME
        warning('minphase-stmcb failed: %s', ME.message);
    end

    % ---- 方法 3: yulewalk ----
    try
        design_fn = @(Rt, wg, ord, fs_) design_matched_iir_yulewalk(Rt, wg, ord);
        [b3, a3, aopt3, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'yulewalk-stmcb');
        warp_infos(end+1) = winfo;
        methods{end+1} = sprintf('yulewalk-stmcb (a=%.4f)', aopt3);
        tmp = evaluate_method(methods{end}, b3, a3, R_target, w_grid, fs);
        if isempty(results), results = tmp; else, results(end+1) = tmp; end
    catch ME
        warning('yulewalk failed: %s', ME.message);
    end

    % ---- 方法 4: iirlpnorm ----
    try
        design_fn = @(Rt, wg, ord, fs_) design_matched_iir_iirlpnorm(Rt, wg, ord);
        [b4, a4, aopt4, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'iirlpnorm');
        warp_infos(end+1) = winfo;
        methods{end+1} = sprintf('iirlpnorm (a=%.4f)', aopt4);
        tmp = evaluate_method(methods{end}, b4, a4, R_target, w_grid, fs);
        if isempty(results), results = tmp; else, results(end+1) = tmp; end
    catch ME
        warning('iirlpnorm failed (maybe DSP System Toolbox missing): %s', ME.message);
    end

    % ---- 方法 5: analog invfreqs + matched discretization ----
    try
        design_fn = @(Rt, wg, ord, fs_) design_matched_iir_analog_matched(Rt, wg, ord, fs_);
        [b5, a5, aopt5, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'analog-invfreqs-matched');
        warp_infos(end+1) = winfo;
        methods{end+1} = sprintf('analog-invfreqs-matched (a=%.4f)', aopt5);
        tmp = evaluate_method(methods{end}, b5, a5, R_target, w_grid, fs);
        if isempty(results), results = tmp; else, results(end+1) = tmp; end
    catch ME
        warning('analog matched failed: %s', ME.message);
    end

    %% =========================
    % 3. 打印结果表
    % ==========================
    fprintf('\n==================== comparison ====================\n');
    fprintf('%-32s | %-12s | %-12s | %-12s | %-10s | %-8s\n', ...
        'method', 'lin-MSE', 'dB-RMSE', 'max|dB|', 'max|p|', 'stable');
    fprintf('%s\n', repmat('-', 1, 104));
    
    for k = 1:numel(results)
        fprintf('%-32s | %-12.4e | %-12.4e | %-12.4e | %-10.6f | %-8d\n', ...
            results(k).name, results(k).mse_linear, results(k).rmse_db, ...
            results(k).max_db_abs, results(k).max_pole_radius, results(k).is_stable);
    end
    
    [~, idx_best_mse] = min([results.mse_linear]);
    [~, idx_best_db ] = min([results.rmse_db]);
    
    fprintf('\nBest by linear MSE : %s\n', results(idx_best_mse).name);
    fprintf('Best by dB RMSE    : %s\n', results(idx_best_db).name);
    
    %% =========================
    % 3b. 打印 warp 优化表
    % ==========================
    fprintf('\n==================== warp optimization summary ====================\n');
    fprintf('%-24s | %-8s | %-8s | %-12s | %-12s | %-10s | %-10s\n', ...
        'method', 'a_init', 'a_final', 'RMSE0(dB)', 'RMSE1(dB)', 'drop', 'drop(%%)');
    fprintf('%s\n', repmat('-', 1, 106));
    
    for k = 1:numel(warp_infos)
        fprintf('%-24s | %-8.4f | %-8.4f | %-12.4e | %-12.4e | %-10.4e | %-9.2f\n', ...
            warp_infos(k).method, ...
            warp_infos(k).a_initial, ...
            warp_infos(k).a_final, ...
            warp_infos(k).rmse_db_initial, ...
            warp_infos(k).rmse_db_final, ...
            warp_infos(k).rmse_db_drop, ...
            warp_infos(k).rmse_db_drop_pct);
    end
    
    %% =========================
    % 3c. 打印更完整的误差改变量
    % ==========================
    fprintf('\n==================== error reduction detail ====================\n');
    fprintf('%-24s | %-12s | %-12s | %-12s | %-12s\n', ...
        'method', 'score_drop', 'score_drop%', 'mse_drop', 'mse_drop%');
    fprintf('%s\n', repmat('-', 1, 84));
    
    for k = 1:numel(warp_infos)
        fprintf('%-24s | %-12.4e | %-12.2f | %-12.4e | %-12.2f\n', ...
            warp_infos(k).method, ...
            warp_infos(k).score_drop, ...
            warp_infos(k).score_drop_pct, ...
            warp_infos(k).mse_drop, ...
            warp_infos(k).mse_drop_pct);
    end

    %% =========================
    % 4. 绘图
    % ==========================
    figure('Name', 'Magnitude response compare');
    semilogx(freqs, 20*log10(mag_target + 1e-15), 'k--', 'LineWidth', 2); hold on;

    legends = {'Analog Prototype'};
    for k = 1:numel(results)
        [h, fplot] = freqz(results(k).b, results(k).a, 16384, fs);
        semilogx(fplot, 20*log10(abs(h) + 1e-15), 'LineWidth', 1.3);
        legends{end+1} = results(k).name; %#ok<AGROW>
    end
    grid on;
    xlim([20, 24000]);
    ylim([-40, 20]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    title('Warp-assisted IIR fitting comparison');
    legend(legends, 'Location', 'best');

    figure('Name', 'Magnitude error compare');
    for k = 1:numel(results)
        [h, ~] = freqz(results(k).b, results(k).a, w_grid);
        err_db = 20*log10(abs(h) + 1e-15) - 20*log10(sqrt(R_target) + 1e-15);
        semilogx(freqs, err_db, 'LineWidth', 1.2); hold on;
    end
    yline(0, 'k--');
    grid on;
    xlim([100, 24000]);
    xlabel('Frequency (Hz)');
    ylabel('Error (dB)');
    title('Magnitude error vs target');
    legend(methods, 'Location', 'best');

    figure('Name', 'Pole-zero maps');
    nshow = numel(results);
    nrow = ceil(nshow / 2);
    for k = 1:nshow
        subplot(nrow, 2, k);
        zplane(results(k).b, results(k).a);
        title(results(k).name);
    end
end

%% ============================================================
% 方法 1：你的谱分解 IRLS
% =============================================================
function [b_final, a_final] = design_matched_iir_spectral(R_target, w_grid, order)
    N = order;
    num_points = length(w_grid);

    num_iters = 8;
    W = ones(num_points, 1);

    for iter = 1:num_iters
        A = zeros(num_points, 2*N + 1);
        A(:, 1) = W;

        for k = 1:N
            A(:, k+1)     = W .* cos(k * w_grid);
            A(:, k+N+1)   = -W .* R_target .* cos(k * w_grid);
        end

        rhs = W .* R_target;
        x = A \ rhs;

        d_temp = [1; x(N+2 : 2*N+1)];
        Q_vals = d_temp(1) * ones(size(w_grid));
        for k = 1:N
            Q_vals = Q_vals + d_temp(k+1) * cos(k * w_grid);
        end

        % 可改成更 aggressive 的重加权
        W = 1 ./ max(abs(Q_vals), 1e-6);
        W = W / mean(W);
    end

    c = x(1 : N+1);
    d = [1; x(N+2 : end)];

    b_unscaled = spectral_factorization(c);
    a_final    = spectral_factorization(d);
    a_final    = stabilize_a(a_final);

    b_final = calibrate_gain(b_unscaled, a_final, sqrt(R_target), w_grid);
    
end

function factor_poly = spectral_factorization(coeffs)
    N = length(coeffs) - 1;
    poly_z = zeros(2*N + 1, 1);

    poly_z(N + 1) = coeffs(1);
    for k = 1:N
        poly_z(N + 1 - k) = coeffs(k+1) / 2;
        poly_z(N + 1 + k) = coeffs(k+1) / 2;
    end

    r = roots(poly_z);

    % 对单位圆外根做镜像，优先构造最小相位/稳定因子
    selected = [];
    used = false(size(r));
    tol = 1e-7;

    for i = 1:length(r)
        if used(i), continue; end
        ri = r(i);

        if abs(abs(ri) - 1) < tol
            selected(end+1,1) = ri; %#ok<AGROW>
            used(i) = true;
            continue;
        end

        target = 1 / conj(ri);
        j = find(~used & abs(r - target) < 1e-4, 1);

        if isempty(j)
            if abs(ri) < 1
                selected(end+1,1) = ri; %#ok<AGROW>
            else
                selected(end+1,1) = 1 / conj(ri); %#ok<AGROW>
            end
            used(i) = true;
        else
            if abs(ri) <= 1
                selected(end+1,1) = ri; %#ok<AGROW>
            else
                selected(end+1,1) = r(j); %#ok<AGROW>
            end
            used([i j]) = true;
        end
    end

    if numel(selected) < N
        [~, idx] = sort(abs(r));
        selected = r(idx(1:N));
    elseif numel(selected) > N
        [~, idx] = sort(abs(selected));
        selected = selected(idx(1:N));
    end

    factor_poly = real(poly(selected(:).'));
    factor_poly = factor_poly / factor_poly(1);
end

%% ============================================================
% 方法 2：最小相位恢复 + invfreqz
% =============================================================
function [b, a] = design_matched_iir_invfreqz(R_target, w_grid, order)
    mag_target = sqrt(max(R_target, 0));

    H_min = reconstruct_minphase_from_mag(mag_target, w_grid);

    wt = build_weight_curve(mag_target, w_grid);
    [b, a] = invfreqz(H_min, w_grid, order, order, wt, 50);

    a = stabilize_a(a);
    b = calibrate_gain(b, a, mag_target, w_grid);
end

function H_min = reconstruct_minphase_from_mag(mag_target, w_grid)
    mag_target = mag_target(:);
    w_grid = w_grid(:);

    n_fft = max(2^18, 2^nextpow2(length(w_grid) * 16));

    w_linear = linspace(0, pi, n_fft/2 + 1)';
    mag_linear = interp1(w_grid, mag_target, w_linear, 'pchip', 'extrap');
    mag_linear = max(mag_linear, 1e-12);

    log_mag_spec = [log(mag_linear); log(mag_linear(end-1:-1:2))];
    cep = ifft(log_mag_spec, 'symmetric');

    win = [1; 2*ones(n_fft/2-1,1); 1; zeros(n_fft/2-1,1)];
    min_phase_spec = exp(fft(cep .* win));

    H_half = min_phase_spec(1:n_fft/2+1);
    H_min = interp1(w_linear, H_half, w_grid, 'pchip', 'extrap');
end


%% ============================================================
% 方法 3/4：prony / stmcb
% =============================================================

function [b, a] = design_matched_iir_prony_minphase(R_target, w_grid, order)
    mag_target = sqrt(max(R_target, 0));

    % 先恢复最小相位复频响
    H_min = reconstruct_minphase_from_mag(mag_target, w_grid);

    % 频响 -> 线性频率网格 -> IFFT 得到冲激响应
    n_fft = max(2^14, 2^nextpow2(length(w_grid) * 4));
    w_linear = linspace(0, pi, n_fft/2 + 1)';
    H_linear = interp1(w_grid, H_min, w_linear, 'pchip', 'extrap');

    % 构造共轭对称频谱
    H_full = [H_linear; conj(H_linear(end-1:-1:2))];
    h = real(ifft(H_full));

    % 截取前一段脉冲响应做 Prony
    imp_len = max(8 * order, 128);
    imp_len = min(imp_len, length(h));
    h_trunc = h(1:imp_len);

    [b, a] = prony(h_trunc, order, order);

    a = stabilize_a(a);
    b = calibrate_gain(b, a, mag_target, w_grid);
end

function [b, a] = design_matched_iir_stmcb_minphase(R_target, w_grid, order)
    mag_target = sqrt(max(R_target, 0));

    % 先恢复最小相位复频响
    H_min = reconstruct_minphase_from_mag(mag_target, w_grid);

    % 频响 -> 冲激响应
    n_fft = max(2^14, 2^nextpow2(length(w_grid) * 4));
    w_linear = linspace(0, pi, n_fft/2 + 1)';
    H_linear = interp1(w_grid, H_min, w_linear, 'pchip', 'extrap');

    H_full = [H_linear; conj(H_linear(end-1:-1:2))];
    h = real(ifft(H_full));

    imp_len = max(8 * order, 128);
    imp_len = min(imp_len, length(h));
    h_trunc = h(1:imp_len);

    % 用 prony 结果做初始化，再 stmcb 迭代细化
    [b0, a0] = prony(h_trunc, order, order);
    [b, a] = stmcb(h_trunc, order, order, 20, a0);

    a = stabilize_a(a);
    b = calibrate_gain(b, a, mag_target, w_grid);
end



%% ============================================================
% 方法 5：yulewalk
% =============================================================
function [b, a] = design_matched_iir_yulewalk(R_target, w_grid, order)
    % 改进版 yulewalk 路线：
    % 1) 对目标幅度做温和平滑
    % 2) 转成功率谱模板给 yulewalk
    % 3) 用 yulewalk 结果初始化 stmcb 做 ARMA 精修
    % 4) 稳定化 + 加权增益校准

    mag_target = sqrt(max(R_target(:), 0));
    w_grid = w_grid(:);

    % yulewalk 频率轴需要 [0,1]
    f_norm = w_grid / pi;

    % ------------------------------------------------------------
    % 1. 构造更适合 yulewalk 的平滑模板
    % ------------------------------------------------------------
    % yulewalk 更适合功率谱模板；对 log-magnitude 做轻微平滑更稳
    logmag = log(max(mag_target, 1e-8));

    % Savitzky-Golay / 移动平均都可以，这里用简单稳妥的平滑
    % 窗长按网格长度自适应，且保持奇数
    win_len = max(31, 2*floor(length(logmag)/200) + 1);
    if mod(win_len, 2) == 0
        win_len = win_len + 1;
    end

    logmag_s = smoothdata(logmag, 'sgolay', win_len);
    mag_s = exp(logmag_s);

    % yulewalk 更适合适度压缩后的模板点，而不是过密点
    n_knots = min(256, max(64, round(length(f_norm)/24)));
    f2 = linspace(0, 1, n_knots).';
    mag2 = interp1(f_norm, mag_s, f2, 'pchip', 'extrap');

    % 关键：yulewalk 从“功率响应”角度更自然，喂 |H|^2
    pwr2 = max(mag2.^2, 1e-12);

    % ------------------------------------------------------------
    % 2. yulewalk 初始解
    % ------------------------------------------------------------
    [b0, a0] = yulewalk(order, f2.', pwr2.');

    % 先稳定化一次
    a0 = stabilize_a(a0);

    % ------------------------------------------------------------
    % 3. 用 yulewalk 初值 + stmcb 精修
    % ------------------------------------------------------------
    % 为了让 stmcb 更贴近目标，先从平滑目标恢复一个近似最小相位复频响，
    % 再变成脉冲响应，最后做 ARMA 迭代精修
    H_seed = reconstruct_minphase_from_mag(mag_s, w_grid);

    n_fft = max(2^14, 2^nextpow2(length(w_grid) * 4));
    w_lin = linspace(0, pi, n_fft/2 + 1).';
    H_lin = interp1(w_grid, H_seed, w_lin, 'pchip', 'extrap');

    H_full = [H_lin; conj(H_lin(end-1:-1:2))];
    h = real(ifft(H_full));

    % 截断到适合低阶 ARMA 的长度
    imp_len = max(128, 12 * order);
    imp_len = min(imp_len, length(h));
    h_trunc = h(1:imp_len);

    % 用 yulewalk 的分母初始化 stmcb
    try
        [b, a] = stmcb(h_trunc, order, order, 25, a0);
    catch
        % 某些版本 stmcb 参数兼容性差，退回 yulewalk 初值
        b = b0;
        a = a0;
    end

    % ------------------------------------------------------------
    % 4. 稳定化
    % ------------------------------------------------------------
    a = stabilize_a(a);

    % ------------------------------------------------------------
    % 5. 加权增益校准
    % ------------------------------------------------------------
    b = calibrate_gain(b, a, mag_target, w_grid);
end


%% ============================================================
% 方法 6：iirlpnorm
% =============================================================
function [b, a] = design_matched_iir_iirlpnorm(R_target, w_grid, order)
    mag_target = sqrt(max(R_target(:), 0));
    w_grid = w_grid(:);
    f_norm = w_grid / pi;

    % ------------------------------------------------------------
    % 1) 平滑目标
    % ------------------------------------------------------------
    logmag = log(max(mag_target, 1e-8));

    win_len = max(21, 2*floor(numel(logmag)/150) + 1);
    if mod(win_len, 2) == 0
        win_len = win_len + 1;
    end

    try
        logmag_s = smoothdata(logmag, 'sgolay', win_len);
    catch
        logmag_s = smoothdata(logmag, 'movmean', win_len);
    end
    %logmag_s = logmag;

    mag_s = exp(logmag_s);

    % ------------------------------------------------------------
    % 2) 压缩为较少设计点
    % ------------------------------------------------------------
    n_knots = min(96, max(32, 10 * order));
    f2 = linspace(0, 1, n_knots).';
    a2 = interp1(f_norm, mag_s, f2, 'pchip', 'extrap');

    floor_val = max(max(a2) * 1e-4, 1e-6);
    a2 = max(a2, floor_val);
    a2(1)   = max(a2(1), floor_val);
    a2(end) = max(a2(end), floor_val);

    % ------------------------------------------------------------
    % 3) 权重
    % ------------------------------------------------------------
    wt = ones(size(a2));
    wt(f2 < 0.05) = 2.0;
    wt(f2 > 0.75) = 1.5;

    slope = [0; abs(diff(log(max(a2, 1e-8))))];
    if max(slope) > 0
        wt = wt .* (1 + 1.5 * slope / max(slope));
    end

    % ------------------------------------------------------------
    % 4) iirlpnorm
    %    用“更宽松”的策略：先尽量给初值；不行就直接裸跑
    % ------------------------------------------------------------
    edges = [0 1];
    p = [2 32];
    dens = 20;

    use_init = true;
    b0 = [];
    a0 = [];

    % 关键修复：
    % reconstruct_minphase_from_mag 的输入长度必须匹配，
    % 所以要在原始 w_grid 上恢复，再插值到 f2*pi
    try
        H_seed_dense = reconstruct_minphase_from_mag(mag_s, w_grid);
        H_seed = interp1(w_grid, H_seed_dense, f2*pi, 'pchip', 'extrap');

        [b0, a0] = invfreqz(H_seed, f2*pi, order, order, wt, 30);

        if any(~isfinite(b0)) || any(~isfinite(a0)) || abs(a0(1)) < 1e-14
            use_init = false;
        else
            b0 = real(b0 / a0(1));
            a0 = real(a0 / a0(1));
            a0 = stabilize_a(a0);
        end
    catch
        use_init = false;
    end

    % 更宽松：有初值就用初值，没有就直接跑
    if use_init
        [b, a] = iirlpnorm(order, order, f2.', edges, a2.', wt.', p, dens, b0, a0);
    else
        [b, a] = iirlpnorm(order, order, f2.', edges, a2.', wt.', p, dens);
    end

    % ------------------------------------------------------------
    % 5) 后处理（也尽量宽松）
    % ------------------------------------------------------------
    if any(~isfinite(b)) || any(~isfinite(a)) || abs(a(1)) < 1e-14
        error('iirlpnorm returned invalid coefficients.');
    end

    b = real(b / a(1));
    a = real(a / a(1));

    % 稳定化，但不要太苛刻
    a = stabilize_a(a);

    % 重新按原目标校准增益
    b = calibrate_gain_passband_ls(b, a, mag_target, w_grid);

    % 最后只做基本合法性检查
    h = freqz(b, a, w_grid);
    if any(~isfinite(h))
        error('iirlpnorm produced non-finite frequency response.');
    end
end

%% ============================================================
% 方法 7：模拟域 invfreqs + matched discretization
% =============================================================
function [b, a] = design_matched_iir_analog_matched(R_target, w_grid, order, fs)
    % 更稳健版 analog 方法：
    % 1) BLT 反扭曲到模拟频率轴
    % 2) 用自由共轭零极点对参数化 4 阶模拟有理函数
    % 3) 多起点 + 正则化拟合模拟域幅度
    % 4) bilinear 回数字域
    % 5) 结果验收，失败则 fallback

    mag_target = sqrt(max(R_target(:), 0));
    w_grid = w_grid(:);

    if order ~= 4
        %warning('This analog implementation is specialized for order = 4. Current order = %d', order);
    end

    % ------------------------------------------------------------
    % 1) BLT 反扭曲 + 高频软限制
    % ------------------------------------------------------------
    Omega_raw = 2 * fs * tan(w_grid / 2);

    % Nyquist 附近会非常大，优化很容易被拖飞；做一个软上限更稳
    Omega_cap = 2 * pi * fs * 8;   % 可调：4~12 倍 fs 都行
    Omega = min(Omega_raw, Omega_cap);

    % ------------------------------------------------------------
    % 2) 找主变化区
    % ------------------------------------------------------------
    mag_db = 20 * log10(mag_target + 1e-12);
    dmag = abs(gradient(mag_db) ./ max(gradient(log(Omega + 10)), 1e-12));
    [~, idx0] = max(dmag);
    idx0 = max(2, min(length(Omega)-1, idx0));
    w0 = Omega(idx0);
    w0 = max(w0, 2*pi*20);

    % 额外找一个“高权重中心频率”
    [~, idx_pk] = max(mag_target);
    wpk = Omega(max(2, min(length(Omega)-1, idx_pk)));
    wpk = max(wpk, 2*pi*20);

    % ------------------------------------------------------------
    % 3) 多起点初始化
    % ------------------------------------------------------------
    gain0 = max(median(mag_target), 1e-3);

    x_candidates = [
        log(gain0), log(0.10*w0), log(0.92*w0), log(0.14*w0), log(1.08*w0), log(0.08*w0), log(0.95*w0), log(0.10*w0), log(1.05*w0);
        log(gain0), log(0.06*w0), log(0.85*w0), log(0.20*w0), log(1.15*w0), log(0.05*w0), log(0.90*w0), log(0.07*w0), log(1.10*w0);
        log(gain0), log(0.12*wpk), log(0.95*wpk), log(0.16*wpk), log(1.05*wpk), log(0.08*wpk), log(0.97*wpk), log(0.09*wpk), log(1.03*wpk);
        log(gain0), log(0.20*w0), log(0.70*w0), log(0.25*w0), log(1.30*w0), log(0.10*w0), log(0.80*w0), log(0.12*w0), log(1.20*w0)
    ];

    obj_scalar = @(x) analog_pz_cost_scalar_regularized(x, Omega, mag_target);

    best_val = inf;
    best_x = x_candidates(1, :);

    opts_nm = optimset( ...
        'Display', 'off', ...
        'MaxIter', 3000, ...
        'MaxFunEvals', 16000, ...
        'TolX', 1e-8, ...
        'TolFun', 1e-9);

    for k = 1:size(x_candidates, 1)
        x0 = x_candidates(k, :);
        try
            x_try = fminsearch(obj_scalar, x0, opts_nm);
            v_try = obj_scalar(x_try);
            if isfinite(v_try) && v_try < best_val
                best_val = v_try;
                best_x = x_try;
            end
        catch
        end
    end

    x1 = best_x;

    % ------------------------------------------------------------
    % 4) 如果有 lsqnonlin，再做向量残差细化
    % ------------------------------------------------------------
    if exist('lsqnonlin', 'file') == 2
        obj_vec = @(x) analog_pz_cost_vector_regularized(x, Omega, mag_target);
        try
            opts = optimoptions('lsqnonlin', ...
                'Display', 'off', ...
                'MaxIterations', 300, ...
                'MaxFunctionEvaluations', 6000, ...
                'FunctionTolerance', 1e-10, ...
                'StepTolerance', 1e-10);
            x1 = lsqnonlin(obj_vec, x1, [], [], opts);
        catch
        end
    end

    % ------------------------------------------------------------
    % 5) 构造模拟域传函
    % ------------------------------------------------------------
    [bs, as] = analog_pz_param_to_tf_safe(x1, w0);

    % 连续域稳定化
    as = local_stabilize_s_den(as);

    % ------------------------------------------------------------
    % 6) 双线性变换
    % ------------------------------------------------------------
    [b, a] = bilinear(bs, as, fs);

    % 数字域稳定化
    a = stabilize_a(a);

    % 增益校准
    b = calibrate_gain_passband_ls(b, a, mag_target, w_grid);

    % ------------------------------------------------------------
    % 7) 结果验收，不合格就 fallback
    % ------------------------------------------------------------
    if ~is_valid_digital_filter(b, a, w_grid)
       % warning('analog matched produced invalid digital filter, fallback to invfreqz.');
        [b, a] = design_matched_iir_invfreqz(R_target, w_grid, order);
        return;
    end

    % 如果误差过大，也认为这次优化跑偏了
    h = freqz(b, a, w_grid);
    err_db = 20*log10(abs(h) + 1e-12) - 20*log10(mag_target + 1e-12);
    if ~all(isfinite(err_db)) || sqrt(mean(err_db.^2)) > 6
       % warning('analog matched fit is poor, fallback to invfreqz.');
        [b, a] = design_matched_iir_invfreqz(R_target, w_grid, order);
        return;
    end
end


function [b, a] = analog_pz_param_to_tf_safe(x, w_ref)
    g = exp(x(1));

    sigma_z1 = exp(x(2)); omega_z1 = exp(x(3));
    sigma_z2 = exp(x(4)); omega_z2 = exp(x(5));
    sigma_p1 = exp(x(6)); omega_p1 = exp(x(7));
    sigma_p2 = exp(x(8)); omega_p2 = exp(x(9));

    % 相对参考频率做更合理的限制
    smin = max(1e-3, 1e-4 * w_ref);
    smax = max(1e2, 20   * w_ref);
    omin = max(1e-2, 0.05 * w_ref);
    omax = max(1e2, 4.00 * w_ref);

    sigma_z1 = min(max(sigma_z1, smin), smax);
    sigma_z2 = min(max(sigma_z2, smin), smax);
    sigma_p1 = min(max(sigma_p1, smin), smax);
    sigma_p2 = min(max(sigma_p2, smin), smax);

    omega_z1 = min(max(omega_z1, omin), omax);
    omega_z2 = min(max(omega_z2, omin), omax);
    omega_p1 = min(max(omega_p1, omin), omax);
    omega_p2 = min(max(omega_p2, omin), omax);

    % 防止两个共轭对频率完全扎堆
    if abs(omega_z2 - omega_z1) < 0.03 * w_ref
        omega_z2 = omega_z1 + 0.03 * w_ref;
    end
    if abs(omega_p2 - omega_p1) < 0.03 * w_ref
        omega_p2 = omega_p1 + 0.03 * w_ref;
    end

    % 最小阻尼，避免超高Q直接飞
    sigma_p1 = max(sigma_p1, 0.02 * omega_p1);
    sigma_p2 = max(sigma_p2, 0.02 * omega_p2);

    z1 = -sigma_z1 + 1j * omega_z1;
    z2 = -sigma_z2 + 1j * omega_z2;
    p1 = -sigma_p1 + 1j * omega_p1;
    p2 = -sigma_p2 + 1j * omega_p2;

    z_all = [z1, conj(z1), z2, conj(z2)];
    p_all = [p1, conj(p1), p2, conj(p2)];

    b = g * real(poly(z_all));
    a = real(poly(p_all));

    b = real_if_close(b);
    a = real_if_close(a);

    if any(~isfinite(b)) || any(~isfinite(a)) || abs(a(1)) < 1e-14
        %error('analog_pz_param_to_tf_safe produced invalid coefficients.');
    end

    b = b / a(1);
    a = a / a(1);
end

function err = analog_pz_cost_vector_regularized(x, Omega, mag_target)
    % 用中频作参考尺度
    w_ref = Omega(max(2, round(0.35 * numel(Omega))));

    try
        [b, a] = analog_pz_param_to_tf_safe(x, w_ref);
    catch
        err = 1e3 * ones(2 * numel(Omega) + 12, 1);
        return;
    end

    p = roots(a);
    z = roots(b);

    if any(~isfinite(p)) || any(~isfinite(z))
        err = 1e3 * ones(2 * numel(Omega) + 12, 1);
        return;
    end

    % 连续域稳定性
    if any(real(p) >= -1e-9)
        err = 1e3 * ones(2 * numel(Omega) + 12, 1);
        return;
    end

    s = 1j * Omega;
    den = polyval(a, s);
    num = polyval(b, s);

    if any(~isfinite(den)) || any(abs(den) < 1e-18) || any(~isfinite(num))
        err = 1e3 * ones(2 * numel(Omega) + 12, 1);
        return;
    end

    H = num ./ den;
    mag = abs(H);

    if any(~isfinite(mag)) || any(mag > 1e6 * max(1, max(mag_target)))
        err = 1e3 * ones(2 * numel(Omega) + 12, 1);
        return;
    end

    err_lin = mag - mag_target;
    err_db  = 20*log10(mag + 1e-9) - 20*log10(mag_target + 1e-9);

    wt = ones(size(Omega));

    % 低频略加强
    wt(Omega < Omega(round(end*0.15))) = 1.3;

    % 高频加强，但别太猛
    wt(Omega > Omega(round(end*0.90))) = wt(Omega > Omega(round(end*0.90))) * 1.6;
    wt(Omega > Omega(round(end*0.97))) = wt(Omega > Omega(round(end*0.97))) * 2.2;

    % 主变化区加权
    gmag = abs(gradient(log(max(mag_target, 1e-9))));
    if max(gmag) > 0
        wt = wt .* (1 + 0.7 * gmag / max(gmag));
    end

    mask_small = mag_target < max(mag_target) * 1e-3;

    wt_db = wt;
    wt_db(mask_small) = 0.12 * wt_db(mask_small);

    wt_lin = wt;
    wt_lin(~mask_small) = 1.8 * wt_lin(~mask_small);

    data_err = [ ...
        0.18 * wt_db  .* err_db; ...
        0.90 * wt_lin .* err_lin ...
    ];

    % ------------------------------------------------------------
    % 正则项
    % ------------------------------------------------------------
    reg = [];

    % 极点阻尼不能太小：sigma / omega 太小意味着Q太高
    p_up = p(imag(p) > 0);
    z_up = z(imag(z) > 0);

    for k = 1:numel(p_up)
        sig = max(-real(p_up(k)), 1e-12);
        omg = max(abs(imag(p_up(k))), 1e-12);
        ratio = sig / omg;
        reg(end+1,1) = 5 * max(0, 0.025 - ratio); %#ok<AGROW>
    end

    % 两个极点对太近时惩罚
    if numel(p_up) >= 2
        dp = abs(imag(p_up(1)) - imag(p_up(2))) / max(w_ref, 1);
        reg(end+1,1) = 2 * max(0, 0.04 - dp); %#ok<AGROW>
    end

    % 两个零点对太近时惩罚
    if numel(z_up) >= 2
        dz = abs(imag(z_up(1)) - imag(z_up(2))) / max(w_ref, 1);
        reg(end+1,1) = 1.2 * max(0, 0.03 - dz); %#ok<AGROW>
    end

    % 极点频率离开主要工作区太远时软惩罚
    for k = 1:numel(p_up)
        omg = abs(imag(p_up(k)));
        reg(end+1,1) = 0.03 * max(0, omg / max(w_ref,1) - 5.0); %#ok<AGROW>
        reg(end+1,1) = 0.10 * max(0, 0.15 - omg / max(w_ref,1)); %#ok<AGROW>
    end

    % 增益过大过小时软惩罚
    lg = x(1);
    reg(end+1,1) = 0.02 * max(0, abs(lg) - 8); %#ok<AGROW>

    err = real([data_err(:); reg(:)]);
end

function val = analog_pz_cost_scalar_regularized(x, Omega, mag_target)
    e = analog_pz_cost_vector_regularized(x, Omega, mag_target);
    val = mean(e.^2);
end

function ok = is_valid_digital_filter(b, a, w_grid)
    ok = true;

    if any(~isfinite(b)) || any(~isfinite(a)) || isempty(a) || abs(a(1)) < 1e-14
        ok = false;
        return;
    end

    p = roots(a);
    if any(~isfinite(p)) || any(abs(p) >= 0.9998)
        ok = false;
        return;
    end

    try
        h = freqz(b, a, w_grid);
    catch
        ok = false;
        return;
    end

    if any(~isfinite(h)) || any(abs(h) > 1e8)
        ok = false;
        return;
    end
end


%% ============================================================
% 工具函数
% =============================================================
function a = stabilize_a(a)
    %[z, p, k] = tf2zp(1, a); %#ok<ASGLU>
    %idx = abs(p) >= 1;
    %p(idx) = 1 ./ conj(p(idx));
    %a = real(poly(p));
    %a = a / a(1);
    a = polystab(a);
end


function wt = build_weight_curve(mag_target, w_grid)
    % 一个比较保守的默认权重
    wt = ones(size(w_grid));

    % 低频稍微加重
    wt(w_grid < 0.08*pi) = 1.5;

    % 峰值区域可适度加重：目标幅度越大，权重略高
    wt = wt .* (0.7 + 0.3 * mag_target / max(mag_target + 1e-12));

    wt = wt(:);
end


function b = calibrate_gain_weighted_ls(b, a, mag_target, w_grid)
    h = freqz(b, a, w_grid);
    % 这里继续保留“平均线性幅度”校准
    g = mean(mag_target) / (mean(abs(h)) + 1e-12);
    b = real(b * g);
end
function b = calibrate_gain(b, a, mag_target, w_grid)
    h = freqz(b, a, w_grid);
    mag_cur = abs(h);

    wt = ones(size(w_grid));
    wt(w_grid < 0.08*pi) = 1.5;
    wt = wt .* (0.7 + 0.3 * mag_target / max(mag_target + 1e-12));

    g = sum(wt .* mag_cur .* mag_target) / (sum(wt .* mag_cur.^2) + 1e-12);
    b = real(b * g);
end

function b = calibrate_gain_passband_ls(b, a, mag_target, w_grid)
    h = freqz(b, a, w_grid);
    mag_cur = abs(h);

    % 只在目标“不是接近 0”的频点上做增益匹配
    mask = mag_target > max(mag_target) * 1e-3;

    % 如果 mask 太少，退化成全带
    if nnz(mask) < 16
        mask = true(size(mag_target));
    end

    mc = mag_cur(mask);
    mt = mag_target(mask);

    % 鲁棒一点：避免极端点支配
    g = sum(mc .* mt) / (sum(mc.^2) + 1e-12);

    b = real(b * g);
end

function [b_best, a_best, a_best_scalar, warp_info] = design_with_warped_target( ...
    design_fn, R_target, w_grid, order, fs, method_name)

    % -------------------------
    % baseline: a = 0
    % -------------------------
    [b0w, a0w] = design_fn(R_target, w_grid, order, fs);
    [b0, a0] = apply_allpass_warp_to_iir(b0w, a0w, 0.0);

    base_eval = evaluate_method(method_name, b0, a0, R_target, w_grid, fs);
    base_score = combined_error_score(base_eval);

    b_best = b0;
    a_best = a0;
    a_best_scalar = 0.0;
    best_eval = base_eval;
    best_score = base_score;

    % -------------------------
    % optimize a
    % -------------------------
    [a_try, score_try, b_try, a_try_tf, eval_try] = optimize_warp_parameter_binary( ...
        design_fn, R_target, w_grid, order, fs, method_name);

    if score_try < best_score
        b_best = b_try;
        a_best = a_try_tf;
        a_best_scalar = a_try;
        best_eval = eval_try;
        best_score = score_try;
    end

    % -------------------------
    % 汇总信息
    % -------------------------
    warp_info = struct();
    warp_info.method = method_name;

    warp_info.a_initial = 0;
    warp_info.a_final   = a_best_scalar;

    warp_info.score_initial = base_score;
    warp_info.score_final   = best_score;
    warp_info.score_drop    = base_score - best_score;
    warp_info.score_drop_pct = 100 * (base_score - best_score) / max(base_score, 1e-12);

    warp_info.rmse_db_initial = base_eval.rmse_db;
    warp_info.rmse_db_final   = best_eval.rmse_db;
    warp_info.rmse_db_drop    = base_eval.rmse_db - best_eval.rmse_db;
    warp_info.rmse_db_drop_pct = 100 * (base_eval.rmse_db - best_eval.rmse_db) / max(base_eval.rmse_db, 1e-12);

    warp_info.mse_initial = base_eval.mse_linear;
    warp_info.mse_final   = best_eval.mse_linear;
    warp_info.mse_drop    = base_eval.mse_linear - best_eval.mse_linear;
    warp_info.mse_drop_pct = 100 * (base_eval.mse_linear - best_eval.mse_linear) / max(base_eval.mse_linear, 1e-20);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%优化器的优化器实现：

function [a_best, best_score, b_best, a_best_tf, best_eval] = optimize_warp_parameter_binary2( ...
    design_fn, R_target, w_grid, order, fs, method_name)

    left = -0.99;
    right = 0.99;
    n_iter = 7;

    a_best = 0.0;
    best_score = inf;
    b_best = [];
    a_best_tf = [];
    best_eval = [];

    for it = 1:n_iter
        m1 = left + 0.35 * (right - left);
        m2 = left + 0.65 * (right - left);

        [score1, b1, a1, eval1] = eval_one_warp_candidate(design_fn, R_target, w_grid, order, fs, method_name, m1);
        [score2, b2, a2, eval2] = eval_one_warp_candidate(design_fn, R_target, w_grid, order, fs, method_name, m2);

        if score1 < best_score
            best_score = score1;
            a_best = m1;
            b_best = b1;
            a_best_tf = a1;
            best_eval = eval1;
        end

        if score2 < best_score
            best_score = score2;
            a_best = m2;
            b_best = b2;
            a_best_tf = a2;
            best_eval = eval2;
        end

        if score1 <= score2
            right = m2;
        else
            left = m1;
        end
    end

    amid = 0.5 * (left + right);
    [scorem, bm, am, evalm] = eval_one_warp_candidate(design_fn, R_target, w_grid, order, fs, method_name, amid);
    if scorem < best_score
        best_score = scorem;
        a_best = amid;
        b_best = bm;
        a_best_tf = am;
        best_eval = evalm;
    end
end

function [a_best, best_score, b_best, a_best_tf, best_eval] = optimize_warp_parameter_binary( ...
    design_fn, R_target, w_grid, order, fs, method_name)

    % ============================================================
    % Extreme warp-parameter optimizer
    %
    % Strategy:
    %   1) Search in u-space, with a = tanh(u)
    %   2) Coarse structured scan in u-space
    %   3) Pick best local basin
    %   4) Golden-section refinement in u-space
    %   5) Small local polish around the best point
    %
    % Designed for:
    %   - expensive eval_one_warp_candidate(...)
    %   - noisy / mildly multi-modal score(a)
    %   - a in (-1, 1), with strong edge sensitivity near +-1
    %
    % Typical eval count:
    %   around 14 ~ 20, configurable below
    % ============================================================

    % ---- search settings ----
    max_evals      = 20;     % total expensive eval budget
    u_bound        = 10;    % tanh(2.8) ~= 0.9926
    golden_iters   = 10;     % refinement iterations (1 eval / iter max)
    do_final_polish = true;  % 2 extra local checks if budget allows

    % ---- outputs ----
    a_best = 0.0;
    best_score = inf;
    b_best = [];
    a_best_tf = [];
    best_eval = [];

    % ---- cache ----
    u_cache = [];
    a_cache = [];
    f_cache = [];
    b_cache = {};
    atf_cache = {};
    eval_cache = {};

    % ============================================================
    % cached evaluator in u-space
    % ============================================================
    function [f, b_cur, a_tf_cur, eval_cur, a_cur] = eval_cached_u(u)
        % clamp u to search range
        u = max(min(u, u_bound), -u_bound);
        a_cur = tanh(u);

        % reuse cached result if same u already evaluated
        if ~isempty(u_cache)
            idx = find(abs(u_cache - u) <= 1e-14, 1);
            if ~isempty(idx)
                f        = f_cache(idx);
                b_cur    = b_cache{idx};
                a_tf_cur = atf_cache{idx};
                eval_cur = eval_cache{idx};
                a_cur    = a_cache(idx);
                return;
            end
        end

        [f, b_cur, a_tf_cur, eval_cur] = eval_one_warp_candidate( ...
            design_fn, R_target, w_grid, order, fs, method_name, a_cur);

        if ~isfinite(f)
            f = inf;
        end

        u_cache(end+1)       = u;
        a_cache(end+1)       = a_cur;
        f_cache(end+1)       = f;
        b_cache{end+1}       = b_cur;
        atf_cache{end+1}     = a_tf_cur;
        eval_cache{end+1}    = eval_cur;

        if f < best_score
            best_score = f;
            a_best = a_cur;
            b_best = b_cur;
            a_best_tf = a_tf_cur;
            best_eval = eval_cur;
        end
    end

    % ============================================================
    % helper: count evals
    % ============================================================
    function n = n_evals()
        n = numel(f_cache);
    end

    % ============================================================
    % 1) coarse structured scan in u-space
    %
    % symmetric but stretched enough to touch near-edge behavior
    %   u = [-2.0, -0.6, 0, 0.6, 2.0]
    %   a = [-0.964, -0.537, 0, 0.537, 0.964]
    % ============================================================
    u0 = [-2.0, -0.6, 0.0, 0.6, 2.0];
    f0 = inf(size(u0));

    for k = 1:numel(u0)
        if n_evals() >= max_evals
            return;
        end
        [f0(k), ~, ~, ~, ~] = eval_cached_u(u0(k));
    end

    % ============================================================
    % 2) choose a basin to refine
    %
    % We want a bracket [ul, ur] around a promising local region.
    % If best point is at the edge, we add one more point outward/inward
    % to avoid refining a bad boundary artifact.
    % ============================================================
    [~, idx_best0] = min(f0);

    if idx_best0 == 1
        % best at left edge: add one point between u0(1) and u0(2)
        u_extra = 0.5 * (u0(1) + u0(2));
        if n_evals() < max_evals
            [f_extra, ~, ~, ~, ~] = eval_cached_u(u_extra);
            if f_extra < f0(1)
                ul = -u_bound;
                ur = u0(2);
            else
                ul = u0(1);
                ur = u0(2);
            end
        else
            ul = u0(1);
            ur = u0(2);
        end

    elseif idx_best0 == numel(u0)
        % best at right edge: add one point between u0(end-1) and u0(end)
        u_extra = 0.5 * (u0(end-1) + u0(end));
        if n_evals() < max_evals
            [f_extra, ~, ~, ~, ~] = eval_cached_u(u_extra);
            if f_extra < f0(end)
                ul = u0(end-1);
                ur = u_bound;
            else
                ul = u0(end-1);
                ur = u0(end);
            end
        else
            ul = u0(end-1);
            ur = u0(end);
        end

    else
        % interior best: use its two neighbors as the initial bracket
        ul = u0(idx_best0 - 1);
        ur = u0(idx_best0 + 1);
    end

    % ============================================================
    % 3) golden-section refinement in u-space
    %
    % robust for noisy objectives, 1 new eval per loop
    % ============================================================
    phi = (sqrt(5) - 1) / 2;   % 0.618...
    x1 = ur - phi * (ur - ul);
    x2 = ul + phi * (ur - ul);

    [f1, ~, ~, ~, ~] = eval_cached_u(x1);
    if n_evals() < max_evals
        [f2, ~, ~, ~, ~] = eval_cached_u(x2);
    else
        return;
    end

    for it = 1:golden_iters
        if n_evals() >= max_evals
            break;
        end

        if f1 <= f2
            ur = x2;
            x2 = x1;
            f2 = f1;

            x1 = ur - phi * (ur - ul);
            [f1, ~, ~, ~, ~] = eval_cached_u(x1);
        else
            ul = x1;
            x1 = x2;
            f1 = f2;

            x2 = ul + phi * (ur - ul);
            [f2, ~, ~, ~, ~] = eval_cached_u(x2);
        end
    end

    % ============================================================
    % 4) optional local polish around current best in u-space
    %
    % this helps when best point lies on a shallow noisy plateau
    % ============================================================
    if do_final_polish && n_evals() < max_evals
        % locate the current best u in cache
        [~, idxg] = min(f_cache);
        u_star = u_cache(idxg);

        % local step based on current bracket width
        du = 0.15 * max(ur - ul, 0.05);

        if n_evals() < max_evals
            eval_cached_u(u_star - du);
        end
        if n_evals() < max_evals
            eval_cached_u(u_star + du);
        end
    end

    % done: global best already tracked in cache
end
%%%%%%%%%%%%%%%%%%%%%%
function [score, b_final, a_final, eval_out] = eval_one_warp_candidate( ...
    design_fn, R_target, w_grid, order, fs, method_name, awarp)

    try
        % 1) 用与最终结构替换一致的频率映射
        [w_map, idx_sort] = warp_frequency_grid(w_grid, awarp);

        % 2) 目标也必须按同一映射重排
        R_warped = R_target(idx_sort);

        % 3) 在 warped 频率轴上设计 G(z)
        [b_warped, a_warped] = design_fn(R_warped, w_map, order, fs);

        % 4) 再做结构级反扭曲：H(z)=G((a-z)/(1-az))
        [b_final, a_final] = apply_allpass_warp_to_iir(b_warped, a_warped, awarp);

        % 5) 回到原始目标上评估
        eval_out = evaluate_method(sprintf('%s(a=%.4f)', method_name, awarp), ...
            b_final, a_final, R_target, w_grid, fs);

        score = combined_error_score(eval_out);

        if ~isfinite(score)
            score = inf;
        end
    catch
        score = inf;
        b_final = [];
        a_final = [];
        eval_out = [];
    end
end

%% ============================================================
% 评估
% =============================================================
function out = evaluate_method(name, b, a, R_target, w_grid, fs)
    mag_target = sqrt(max(R_target, 0));
    h = freqz(b, a, w_grid);
    mag = abs(h);

    err_lin = mag - mag_target;
    err_db  = 20*log10(mag + 1e-15) - 20*log10(mag_target + 1e-15);

    p = roots(a);

    out.name = name;
    out.b = b;
    out.a = a;
    out.mse_linear = mean(err_lin.^2);
    out.rmse_db = sqrt(mean(err_db.^2));
    out.max_db_abs = max(abs(err_db));
    out.max_pole_radius = max(abs(p));
    out.is_stable = all(abs(p) < 1 - 1e-10);

    [out.h_plot, out.f_plot] = freqz(b, a, 4096, fs);
end
function s = combined_error_score(out)
    s = out.rmse_db ...
      + 0.15 * out.max_db_abs ...
      + 0.02 * sqrt(out.mse_linear);

    if ~out.is_stable
        s = s + 1e3;
    end

    if out.max_pole_radius >= 0.9995
        s = s + 50;
    end
end

%%
function [w_map_sorted, idx_sort] = warp_frequency_grid(w_grid, a)
    % 与最终结构替换完全一致的映射：
    % z2 = (a - z) / (1 - a z), z = e^{jw}
    %
    % 这个映射会把 w=0 映到 w2=pi，把 w=pi 映到 w2=0，
    % 也就是“低频 -> 高频”。
    %
    % 由于多数设计函数要求频率轴单调递增，
    % 所以这里把映射后的频率排序，并返回排序下标。

    z = exp(1j * w_grid(:));
    z2 = (a - z) ./ (1 - a * z);

    % z2 在单位圆上，频率取 [0, pi]
    % 用 acos(real(z2)) 比 angle 更稳，且天然落在 [0, pi]
    w_map = acos(max(-1, min(1, real(z2))));

    % 排序成升序，供 design_fn 使用
    [w_map_sorted, idx_sort] = sort(w_map, 'ascend');

    % 避免重复点/非严格单调
    for k = 2:length(w_map_sorted)
        if w_map_sorted(k) <= w_map_sorted(k-1)
            w_map_sorted(k) = min(pi, w_map_sorted(k-1) + 1e-12);
        end
    end
end

function [b_out, a_out] = apply_allpass_warp_to_iir(b_in, a_in, a)
    % 将设计域滤波器 G(z) 变成 H(z) = G((a-z)/(1-az))
    % 在 q = z^-1 多项式域下：
    % q2 = (a - q) / (1 - a q)

    b_in = real_if_close(b_in(:).');
    a_in = real_if_close(a_in(:).');

    Mb = length(b_in) - 1;
    Ma = length(a_in) - 1;
    L = max(Mb, Ma);

    b_pad = [b_in, zeros(1, L - Mb)];
    a_pad = [a_in, zeros(1, L - Ma)];

    % q2 = (a - q)/(1 - a q)
    p_num = [a, -1];   % a - q
    p_den = [1, -a];   % 1 - a q

    num_acc = 0;
    den_acc = 0;

    for k = 0:L
        term_num = poly_power_asc(p_num, k);
        term_den = poly_power_asc(p_den, L-k);
        term = conv(term_num, term_den);

        num_acc = poly_add_asc(num_acc, b_pad(k+1) * term);
        den_acc = poly_add_asc(den_acc, a_pad(k+1) * term);
    end

    b_out = real_if_close(num_acc);
    a_out = real_if_close(den_acc);

    % 归一化
    if abs(a_out(1)) < 1e-14
        error('Warped denominator leading coefficient is too small.');
    end

    b_out = b_out / a_out(1);
    a_out = a_out / a_out(1);

    a_out = stabilize_a(a_out);

    % 再做一次实数化
    b_out = real_if_close(b_out);
    a_out = real_if_close(a_out);
end

function p = poly_power_asc(base, n)
    % base 用“升幂”表示：c0 + c1 q + c2 q^2 + ...
    if n == 0
        p = 1;
        return;
    end

    p = 1;
    for k = 1:n
        p = conv(p, base);
    end
end

function c = poly_add_asc(a, b)
    if isequal(a, 0), c = b; return; end
    if isequal(b, 0), c = a; return; end

    la = length(a);
    lb = length(b);
    L = max(la, lb);

    aa = [a, zeros(1, L-la)];
    bb = [b, zeros(1, L-lb)];

    c = aa + bb;
end

function x = real_if_close(x)
    if max(abs(imag(x))) < 1e-10 * max(1, max(abs(real(x))))
        x = real(x);
    end
end