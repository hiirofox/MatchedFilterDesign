%% ============================================================
%  Monotonic Omega(w) optimization + rational G(z) fitting
%
%  Step 1:
%     Design monotonic Omega(w) by
%         dOmega/dw = exp(Q(w))
%
%  Step 2:
%     Fit G(z) of the form
%         G(z) = (1/T)*(z - z^-1)*P(c)/Q(c),  c=(z+z^-1)/2
%
%     On unit circle:
%         Omega_G(w) = (2/T)*sin(w)*P(cos w)/Q(cos w)
%
%  Step 3:
%     Use G(z) as substitution for analog prototype:
%         H_z(z) = H_s(G(z))
%
%  Output:
%     - optimized Omega(w)
%     - fitted P/Q coefficients
%     - numerator/denominator of G(z) in x=z^-1 form
%     - frequency-response comparison plots
%% ============================================================

clear; clc; close all;

%% ---------------- User parameters ----------------
fs   = 48000;
T    = 1/fs;

fmin = 20;
fmax = fs/2;
Nw   = 4000;

orderOmega = 32;   % order of Q(w) basis for monotonic Omega design
orderG     = 6;   % order of rational fit P/Q for G(z)

% Optimization options for Omega
maxIterOmega = 12000;
displayOpt   = 'iter';   % 'off' or 'iter'

% Weights for Omega optimization
lambda_hi      = 8;
pow_hi         = 3;
lambda_anchor  = 200;
lambda_smooth  = 1e-2;
lambda_qenergy = 1e-3;

eps_log = 1e-12;

% Rational fitting iterations
nSK = 8;   % Sanathanan-Koerner style iterations for rational fit

% Analog prototype for validation
f0 = 21000;
Qp = 5.707;
w0 = 2*pi*f0;

%% ---------------- Frequency grid ----------------
f = logspace(log10(fmin), log10(fmax), Nw);
w = 2*pi*f/fs;
c = cos(w);
s = sin(w);

Omega_target = w / T;
W = 1 + lambda_hi*(w/pi).^pow_hi;

%% ---------------- Anchor points ----------------
f_anchor = [200, 1000, 3000, 8000, 12000, 18000, 22000];
f_anchor = f_anchor(f_anchor < fs/2);
w_anchor = 2*pi*f_anchor/fs;
Omega_anchor_target = w_anchor / T;

%% ============================================================
%% Step 1: Optimize monotonic Omega(w)
%% ============================================================
q0 = zeros(orderOmega+1,1);

objfun = @(q) objective_monotonic_mapping( ...
    q, w, Omega_target, W, ...
    w_anchor, Omega_anchor_target, ...
    T, lambda_anchor, lambda_smooth, lambda_qenergy, eps_log);

opts = optimset('Display', displayOpt, ...
                'MaxIter', maxIterOmega, ...
                'MaxFunEvals', 200000, ...
                'TolX', 1e-10, ...
                'TolFun', 1e-10);

q_opt = fminsearch(objfun, q0, opts);

[Qw, dOmega_raw, Omega_opt] = build_monotonic_mapping(q_opt, w, T);

%% ============================================================
%% Step 2: Fit rational G(z) with anchored form
%%
%% R(c) = [0.5 + (1-c) Pt(c)] / [1 + (1-c) Qt(c)]
%%
%% Pt(c)=p0+p1*c+...+pm*c^m
%% Qt(c)=q0+q1*c+...+qn*c^n
%%
%% This guarantees:
%%   R(1)=0.5 exactly
%%   denominator at c=1 is exactly 1
%% ============================================================

% IMPORTANT: exclude exact Nyquist from fitting
f_fit = logspace(log10(fmin), log10((fs/2)*(1-1e-6)), Nw);
w_fit = 2*pi*f_fit/fs;
c_fit = cos(w_fit);
s_fit = sin(w_fit);

Omega_fit_target = interp1(w, Omega_opt, w_fit, 'pchip');
W_fit = 1 + lambda_hi*(w_fit/pi).^pow_hi;

R_target = (Omega_fit_target .* T) ./ (2*s_fit);

[pt, qt] = fit_R_anchored_SK(c_fit, R_target, W_fit, orderG, orderG, nSK);

R_fit_eval = eval_R_anchored(c, pt, qt);
Omega_fit = (2/T) * s .* R_fit_eval;

fprintf('\nAnchored fit coefficients:\n');
fprintf('Pt = '); disp(pt);
fprintf('Qt = '); disp(qt);

fprintf('R_fit(1) = %.12f (target 0.5)\n', eval_R_anchored(1, pt, qt));

%% ============================================================
%% Step 3: Use G(z) to transform analog prototype
%% ============================================================
% BLT for comparison
Omega_blt = (2/T) * tan(w/2);

Ns_blt = (2/T) * [1 -1];
Ds_blt = [1 1];
[Bz_blt, Az_blt] = analog2digital_by_substitution(Ns_blt, Ds_blt, w0, Qp);

% Use fitted G(z)
[Ng, Dg] = build_G_polynomials_from_anchored(pt, qt, T);
[Bz_fit, Az_fit] = analog2digital_by_substitution(Ng, Dg, w0, Qp);

%% ---------------- Frequency responses ----------------
Hs = analog_lp_response(2*pi*f, w0, Qp);

Hz_blt_map = analog_lp_response(Omega_blt, w0, Qp);   % mapping-only view
Hz_opt_map = analog_lp_response(Omega_opt, w0, Qp);   % optimized Omega-only
Hz_fit_map = analog_lp_response(Omega_fit, w0, Qp);   % fitted G mapping view

x_eval = exp(-1j*w);
Hz_fit = polyval_asc(Bz_fit, x_eval) ./ polyval_asc(Az_fit, x_eval);
Hz_blt = polyval_asc(Bz_blt, x_eval) ./ polyval_asc(Az_blt, x_eval);

Hs_dB      = 20*log10(abs(Hs) + 1e-15);
Hz_blt_dB  = 20*log10(abs(Hz_blt) + 1e-15);
Hz_fit_dB  = 20*log10(abs(Hz_fit) + 1e-15);
Hz_opt_dB  = 20*log10(abs(Hz_opt_map) + 1e-15);
Hz_fitm_dB = 20*log10(abs(Hz_fit_map) + 1e-15);

fprintf('BLT dB RMSE          = %.6f dB\n', rms(Hz_blt_dB - Hs_dB));
fprintf('Optimized Omega RMSE = %.6f dB\n', rms(Hz_opt_dB - Hs_dB));
fprintf('Fitted G-map RMSE    = %.6f dB\n', rms(Hz_fitm_dB - Hs_dB));
fprintf('Actual H(s)->H(z) RMSE via fitted G = %.6f dB\n', rms(Hz_fit_dB - Hs_dB));

%% ---------------- Plots ----------------
figure('Color','w');
semilogx(f, Omega_target/(2*pi), 'LineWidth', 1.8); hold on;
semilogx(f, Omega_blt/(2*pi), '--', 'LineWidth', 1.4);
semilogx(f, Omega_opt/(2*pi), 'LineWidth', 1.8);
semilogx(f, Omega_fit/(2*pi), 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Digital frequency (Hz)');
ylabel('Mapped analog frequency (Hz)');
title(sprintf('Frequency mapping | orderOmega=%d, orderG=%d', orderOmega, orderG));
legend('Ideal \Omega=\omega/T', 'BLT', 'Optimized \Omega', 'Fitted G', 'Location', 'best');

figure('Color','w');
semilogx(f, Hs_dB, 'LineWidth', 1.8); hold on;
semilogx(f, Hz_blt_dB, '--', 'LineWidth', 1.4);
semilogx(f, Hz_opt_dB, 'LineWidth', 1.8);
semilogx(f, Hz_fit_dB, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
ylim([-40 20]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(sprintf('Analog vs BLT vs optimized \\Omega vs fitted G(z) | f_s=%g, f_0=%g, Q=%g', fs, f0, Qp));
legend('Analog prototype', 'BLT', 'Optimized \Omega only', 'Fitted G(z)', 'Location', 'best');

figure('Color','w');
semilogx(f, Hz_blt_dB - Hs_dB, '--', 'LineWidth', 1.4); hold on;
semilogx(f, Hz_opt_dB - Hs_dB, 'LineWidth', 1.8);
semilogx(f, Hz_fit_dB - Hs_dB, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('Magnitude error (dB)');
title('Magnitude error');
legend('BLT', 'Optimized \Omega only', 'Fitted G(z)', 'Location', 'best');

figure('Color','w');
semilogx(f, numerical_derivative(w, Omega_opt), 'LineWidth', 1.6); hold on;
semilogx(f, numerical_derivative(w, Omega_fit), 'LineWidth', 1.6);
yline(0, '--');
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('d\Omega/d\omega');
title('Monotonicity check');
legend('Optimized \Omega', 'Fitted G', 'Location', 'best');

%% ---------------- Report final substitution ----------------
disp(' ');
disp('Final substitution form:');
disp('G(z) = (1/T)*(z-z^-1)*P(c)/Q(c),  c=(z+z^-1)/2');
disp('P(c) coefficients:');
disp(p);
disp('Q(c) coefficients (with leading 1):');
disp([1 qg]);

disp('Numerator Ng(x), Denominator Dg(x) for x=z^-1:');
disp('Ng ='); disp(Ng);
disp('Dg ='); disp(Dg);

%% ============================================================
%% Local functions
%% ============================================================

function J = objective_monotonic_mapping( ...
    q, w, Omega_target, W, ...
    w_anchor, Omega_anchor_target, ...
    T, lambda_anchor, lambda_smooth, lambda_qenergy, eps_log)

    [Qw, dOmega_raw, Omega] = build_monotonic_mapping(q, w, T);

    Elog = log(Omega + eps_log) - log(Omega_target + eps_log);
    Jfit = mean(W(:) .* (Elog(:).^2));

    Omega_anchor = interp1(w, Omega, w_anchor, 'pchip');
    Eanchor = log(Omega_anchor + eps_log) - log(Omega_anchor_target + eps_log);
    Janchor = mean(Eanchor.^2);

    d2 = numerical_derivative(w, dOmega_raw);
    Jsmooth = mean(d2.^2);

    Jq = mean(q(:).^2);

    J = Jfit ...
      + lambda_anchor  * Janchor ...
      + lambda_smooth  * Jsmooth ...
      + lambda_qenergy * Jq;

    if any(~isfinite(Omega)) || any(~isfinite(dOmega_raw)) || ~isfinite(J)
        J = 1e30;
    end
end

function [Qw, dOmega_raw, Omega] = build_monotonic_mapping(q, w, T)
    Qw = zeros(size(w));
    for k = 0:length(q)-1
        Qw = Qw + q(k+1) * cos(k*w);
    end

    dOmega_raw = exp(Qw);
    Omega_raw = cumtrapz(w, dOmega_raw);

    scale = (pi/T) / Omega_raw(end);
    Omega = scale * Omega_raw;
end
function [pt, qt] = fit_R_anchored_SK(c, r, W, m, n, nIter)
    % Fit:
    %   R(c) = [0.5 + (1-c) Pt(c)] / [1 + (1-c) Qt(c)]
    %
    % Pt(c)=p0+p1*c+...+pm*c^m
    % Qt(c)=q0+q1*c+...+qn*c^n

    pt = zeros(1, m+1);
    qt = zeros(1, n+1);

    for it = 1:nIter
        Qprev = 1 + (1-c).*polyval_desc(fliplr(qt), c);
        wt = sqrt(W(:)) ./ max(abs(Qprev(:)), 1e-8);

        % unknown vector x = [pt(0..m), qt(0..n)]
        A = zeros(length(c), (m+1) + (n+1));
        b = zeros(length(c), 1);

        for i = 1:length(c)
            ci = c(i);
            ri = r(i);

            % Pt terms
            for k = 0:m
                A(i, k+1) = (1-ci) * ci^k;
            end

            % Qt terms
            for k = 0:n
                A(i, (m+1)+k+1) = -ri * (1-ci) * ci^k;
            end

            % move constant terms to rhs:
            % 0.5 + (1-c)Pt - r[1 + (1-c)Qt] = 0
            % => (1-c)Pt - r(1-c)Qt = r - 0.5
            b(i) = ri - 0.5;
        end

        Aw = A .* wt;
        bw = b .* wt;

        x = Aw \ bw;

        pt = x(1:m+1).';
        qt = x(m+2:end).';
    end
end

function R = eval_R_anchored(c, pt, qt)
    Pt = polyval_desc(fliplr(pt), c);
    Qt = polyval_desc(fliplr(qt), c);

    num = 0.5 + (1-c).*Pt;
    den = 1   + (1-c).*Qt;

    R = num ./ den;
end

function [Ng, Dg] = build_G_polynomials_from_anchored(pt, qt, T)
    % Build G(x), x=z^-1
    % G(z) = (1/T)(z-z^-1)R(c), c=(z+z^-1)/2
    %
    % R(c) = [0.5 + (1-c)Pt(c)] / [1 + (1-c)Qt(c)]

    mp = length(pt)-1;
    nq = length(qt)-1;

    % Build Pt(c) and Qt(c) in x-polynomial form
    % x^k c^k --> ((1+x^2)^k)/(2^k)
    Pt_x = 0;
    for k = 0:mp
        base = poly_pow([1 0 1], k);    % (1+x^2)^k, ascending powers
        term = (pt(k+1)/(2^k)) * prepend_zeros(base, mp-k);
        Pt_x = poly_add(Pt_x, term);
    end

    Qt_x = 0;
    for k = 0:nq
        base = poly_pow([1 0 1], k);
        term = (qt(k+1)/(2^k)) * prepend_zeros(base, nq-k);
        Qt_x = poly_add(Qt_x, term);
    end

    % c = (x + x^-1)/2, so in aligned polynomial form:
    % x*c = (1+x^2)/2
    % x*(1-c) = x - (1+x^2)/2 = -(1/2) + x - (1/2)x^2
    one_minus_c_x = [-0.5 1 -0.5];   % ascending powers

    % Numerator of R:
    % 0.5 + (1-c)Pt(c)
    % align powers by multiplying with enough x powers
    NR = poly_add( ...
        0.5 * prepend_zeros(1, mp+1), ...
        conv(one_minus_c_x, Pt_x) );

    % Denominator of R:
    DR = poly_add( ...
        prepend_zeros(1, nq+1), ...
        conv(one_minus_c_x, Qt_x) );

    % Make same alignment length
    L = max(length(NR), length(DR));
    NR = [NR zeros(1, L-length(NR))];
    DR = [DR zeros(1, L-length(DR))];

    % G(x) = (1/T)*(1-x^2)*NR / (x*DR)
    Num = conv([1 0 -1], NR);
    Den = prepend_zeros(DR, 1);   % multiply by x

    Ng = (1/T) * trim_poly(Num);
    Dg = trim_poly(Den);
end


function y = polyval_desc(p, x)
    % p is in descending powers, like MATLAB polyval
    y = polyval(p, x);
end
function [Bz, Az] = analog2digital_by_substitution(Ns, Ds, w0, Q)
    Bz = (w0^2) * conv(Ds, Ds);
    Az = poly_add_many({
        conv(Ns, Ns), ...
        (w0/Q) * conv(Ns, Ds), ...
        (w0^2) * conv(Ds, Ds)
    });

    Bz = trim_poly(Bz);
    Az = trim_poly(Az);

    Bz = Bz / Az(1);
    Az = Az / Az(1);
end

function H = analog_lp_response(Omega, w0, Q)
    s = 1j * Omega;
    H = (w0^2) ./ (s.^2 + (w0/Q)*s + w0^2);
end

function p = poly_pow(a, n)
    p = 1;
    for k = 1:n
        p = conv(p, a);
    end
    p = trim_poly(p);
end

function out = prepend_zeros(p, n)
    if n <= 0
        out = p;
    else
        out = [zeros(1,n), p];
    end
end

function c = poly_add(a, b)
    la = length(a);
    lb = length(b);
    L = max(la, lb);
    c = zeros(1, L);
    c(1:la) = c(1:la) + a;
    c(1:lb) = c(1:lb) + b;
    c = trim_poly(c);
end

function c = poly_add_many(cellpolys)
    c = 0;
    for k = 1:numel(cellpolys)
        c = poly_add(c, cellpolys{k});
    end
    c = trim_poly(c);
end

function y = polyval_asc(p, x)
    y = zeros(size(x));
    xp = ones(size(x));
    for k = 1:length(p)
        y = y + p(k) * xp;
        xp = xp .* x;
    end
end

function p = trim_poly(p)
    tol = 1e-13;
    idx = find(abs(p) > tol, 1, 'last');
    if isempty(idx)
        p = 0;
    else
        p = p(1:idx);
    end
end

function d = numerical_derivative(x, y)
    d = zeros(size(y));
    d(2:end-1) = (y(3:end) - y(1:end-2)) ./ (x(3:end) - x(1:end-2));
    d(1) = (y(2)-y(1)) / (x(2)-x(1));
    d(end) = (y(end)-y(end-1)) / (x(end)-x(end-1));
end