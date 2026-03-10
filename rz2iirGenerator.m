
function rz2iirGenerator()
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

    order = 4;

    %% =========================
    % 1. 定义模拟目标幅度
    % ==========================
    fc = 180;
    wc = 2 * pi * fc;
    Q = 2;
    stages = 4;

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
    Hs_obj = Hs_peaking(s);
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

    % ---- 方法 3: 最小相位 + stmcb ----
    design_fn = @(Rt, wg, ord, fs_) design_matched_iir_stmcb_minphase(Rt, wg, ord);
    [b2c, a2c, aopt2c, winfo] = design_with_warped_target(design_fn, R_target, w_grid, order, fs, 'minphase-stmcb');
    warp_infos(end+1) = winfo;
    methods{end+1} = sprintf('minphase-stmcb (a=%.4f)', aopt2c);
    results(end+1) = evaluate_method(methods{end}, b2c, a2c, R_target, w_grid, fs);
   

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
    xlim([100, 24000]);
    ylim([-60, 30]);
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
% stmcb
% =============================================================

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


function b = calibrate_gain(b, a, mag_target, w_grid)
    h = freqz(b, a, w_grid);
    mag_cur = abs(h);

    wt = ones(size(w_grid));
    wt(w_grid < 0.08*pi) = 1.5;
    wt = wt .* (0.7 + 0.3 * mag_target / max(mag_target + 1e-12));

    g = sum(wt .* mag_cur .* mag_target) / (sum(wt .* mag_cur.^2) + 1e-12);
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


function [a_best, best_score, b_best, a_best_tf, best_eval] = optimize_warp_parameter_binary( ...
    design_fn, R_target, w_grid, order, fs, method_name)

    % ============================================================
    % Mesh search for warp parameter a
    %
    % Stage 1:
    %   Uniformly sample n points in [a_min, a_max]
    %
    % Stage 2:
    %   Find the best sampled point, then refine inside one adjacent interval
    %   around that point using m points
    %
    % Stage 3:
    %   Run local interval shrink ("binary"/ternary-like) in that refined interval
    %
    % Notes:
    %   - lower score is better
    %   - cache is used to avoid duplicate expensive evaluations
    % ============================================================

    % ---------- user-tunable parameters ----------
    a_min = -1.99;
    a_max =  1.99;

    n_coarse = 30;     % first-stage global mesh
    m_fine   = 30;     % second-stage local mesh
    n_local_iter = 24; % final local shrink iterations

    % local shrink probe positions inside interval
    r1 = 0.35;
    r2 = 0.65;

    % ---------- outputs ----------
    a_best = 0.0;
    best_score = inf;
    b_best = [];
    a_best_tf = [];
    best_eval = [];

    % ---------- cache ----------
    a_cache = [];
    f_cache = [];
    b_cache = {};
    atf_cache = {};
    eval_cache = {};

    % ============================================================
    % cached evaluator
    % ============================================================
    function [f, b_cur, a_tf_cur, eval_cur] = eval_cached(a_try)
        % clamp
        a_try = max(min(a_try, a_max), a_min);

        % reuse if already evaluated
        if ~isempty(a_cache)
            idx = find(abs(a_cache - a_try) <= 1e-14, 1);
            if ~isempty(idx)
                f        = f_cache(idx);
                b_cur    = b_cache{idx};
                a_tf_cur = atf_cache{idx};
                eval_cur = eval_cache{idx};
                return;
            end
        end

        [f, b_cur, a_tf_cur, eval_cur] = eval_one_warp_candidate( ...
            design_fn, R_target, w_grid, order, fs, method_name, a_try);

        if ~isfinite(f)
            f = inf;
        end

        a_cache(end+1)    = a_try;
        f_cache(end+1)    = f;
        b_cache{end+1}    = b_cur;
        atf_cache{end+1}  = a_tf_cur;
        eval_cache{end+1} = eval_cur;

        if f < best_score
            best_score = f;
            a_best = a_try;
            b_best = b_cur;
            a_best_tf = a_tf_cur;
            best_eval = eval_cur;
        end
    end

    % ============================================================
    % Stage 1: coarse global mesh
    % ============================================================
    A1 = linspace(a_min, a_max, n_coarse);
    F1 = inf(size(A1));

    for k = 1:numel(A1)
        [F1(k), ~, ~, ~] = eval_cached(A1(k));
    end

    % best coarse point
    [~, idx1] = min(F1);

    % choose one adjacent interval around the best coarse point
    if idx1 == 1
        left1  = A1(1);
        right1 = A1(2);
    elseif idx1 == numel(A1)
        left1  = A1(end-1);
        right1 = A1(end);
    else
        % compare left and right adjacent intervals using neighbor scores
        if F1(idx1-1) <= F1(idx1+1)
            left1  = A1(idx1-1);
            right1 = A1(idx1);
        else
            left1  = A1(idx1);
            right1 = A1(idx1+1);
        end
    end

    % ============================================================
    % Stage 2: fine mesh inside selected interval
    % ============================================================
    A2 = linspace(left1, right1, m_fine);
    F2 = inf(size(A2));

    for k = 1:numel(A2)
        [F2(k), ~, ~, ~] = eval_cached(A2(k));
    end

    [~, idx2] = min(F2);

    % choose local interval around best fine point
    if idx2 == 1
        left2  = A2(1);
        right2 = A2(2);
    elseif idx2 == numel(A2)
        left2  = A2(end-1);
        right2 = A2(end);
    else
        left2  = A2(idx2-1);
        right2 = A2(idx2+1);
    end

    % ============================================================
    % Stage 3: local interval shrink
    % ============================================================
    left = left2;
    right = right2;

    for it = 1:n_local_iter
        m1 = left + r1 * (right - left);
        m2 = left + r2 * (right - left);

        [score1, ~, ~, ~] = eval_cached(m1);
        [score2, ~, ~, ~] = eval_cached(m2);

        if score1 <= score2
            right = m2;
        else
            left = m1;
        end
    end

    % final midpoint check
    amid = 0.5 * (left + right);
    eval_cached(amid);
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

    eps_mag = 1e-12;

    mag_db_target = 20*log10(mag_target + eps_mag);
    mag_db = 20*log10(mag + eps_mag);

    err_lin = mag - mag_target;
    err_db  = mag_db - mag_db_target;

    % ------------------------------------------------------------
    % Visual-focus weighting:
    % emphasize target magnitude above -40 dB
    % ------------------------------------------------------------
    db_focus = -40;   % focus threshold
    db_soft  = 4;     % softness around threshold
    w_floor  = 0.08;  % minimum weight below threshold

    w_vis = w_floor + (1 - w_floor) ./ ...
        (1 + exp(-(mag_db_target - db_focus)/db_soft));

    w_vis = w_vis(:);
    w_vis = w_vis / mean(w_vis);   % normalize average weight to ~1

    % log-frequency axis
    w_safe = max(w_grid(:), 1e-9);
    xlog = log(w_safe);

    % derivatives on log-frequency axis
    d_tar = gradient(mag_db_target(:), xlog);
    d_fit = gradient(mag_db(:), xlog);
    derr1 = d_fit - d_tar;

    dd_tar = gradient(d_tar, xlog);
    dd_fit = gradient(d_fit, xlog);
    derr2 = dd_fit - dd_tar;

    % poles
    p = roots(a);
    pole_radius = abs(p);

    % store basics
    out.name = name;
    out.b = b;
    out.a = a;

    out.err_db = err_db(:);
    out.err_lin = err_lin(:);
    out.mag_db = mag_db(:);
    out.mag_db_target = mag_db_target(:);
    out.w_vis = w_vis;

    % ------------------------------------------------------------
    % weighted error metrics
    % ------------------------------------------------------------
    out.mse_linear = mean(err_lin.^2);
    out.rmse_linear = sqrt(out.mse_linear);

    out.wrmse_db = sqrt(sum(w_vis .* (err_db(:).^2)) / sum(w_vis));
    out.wrmse_lin = sqrt(sum(w_vis .* (err_lin(:).^2)) / sum(w_vis));

    % unweighted versions kept for reference
    out.rmse_db = sqrt(mean(err_db.^2));
    out.max_db_abs = max(abs(err_db));

    % weighted slope / curvature
    out.wrmse_slope_db = sqrt(sum(w_vis .* (derr1.^2)) / sum(w_vis));
    out.wrmse_curv_db  = sqrt(sum(w_vis .* (derr2.^2)) / sum(w_vis));

    % roughness of fitted curve
    out.roughness_db = sqrt(sum(w_vis .* (dd_fit.^2)) / sum(w_vis));

    % ------------------------------------------------------------
    % percentile-like stats only in visually relevant region
    % ------------------------------------------------------------
    idx_vis = (mag_db_target >= -40);

    if any(idx_vis)
        e_vis = abs(err_db(idx_vis));
        out.p95_db_abs_vis = prctile(e_vis, 95);
        out.p98_db_abs_vis = prctile(e_vis, 98);
        out.max_db_abs_vis = max(e_vis);
    else
        e_all = abs(err_db);
        out.p95_db_abs_vis = prctile(e_all, 95);
        out.p98_db_abs_vis = prctile(e_all, 98);
        out.max_db_abs_vis = max(e_all);
    end

    % ------------------------------------------------------------
    % anchor errors in visually meaningful places
    % ------------------------------------------------------------
    n = numel(w_grid);

    idx_lo = max(1, round(0.08 * n));
    idx_hi = max(1, round(0.92 * n));

    [~, idx_peak_tar] = max(mag_db_target);

    out.anchor_lo_db   = abs(err_db(idx_lo));
    out.anchor_peak_db = abs(err_db(idx_peak_tar));
    out.anchor_hi_db   = abs(err_db(idx_hi));

    % stability
    out.max_pole_radius = max(pole_radius);
    out.is_stable = all(pole_radius < 1 - 1e-10);

    [out.h_plot, out.f_plot] = freqz(b, a, 4096, fs);
end
function s = combined_error_score(out, a_warp)
    % Main fit score: prioritize visually relevant region
    s = 1.00 * out.wrmse_db ...
      + 0.30 * out.p95_db_abs_vis ...
      + 0.12 * out.wrmse_lin ...
      + 0.16 * out.wrmse_slope_db ...
      + 0.04 * out.wrmse_curv_db;

    % Anchors
    s = s ...
      + 0.10 * out.anchor_lo_db ...
      + 0.16 * out.anchor_peak_db ...
      + 0.10 * out.anchor_hi_db;

    % Stability
    if ~out.is_stable
        s = s + 1e3;
        return;
    end

    % Pole radius soft penalty
    if out.max_pole_radius >= 0.9995
        s = s + 50;
    elseif out.max_pole_radius >= 0.9990
        s = s + 15;
    elseif out.max_pole_radius >= 0.9985
        s = s + 5;
    end

    % Warp-edge soft penalty
    if nargin >= 2 && ~isempty(a_warp)
        excess = max(0, abs(a_warp) - 0.90);
        s = s + 40 * excess.^2;
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