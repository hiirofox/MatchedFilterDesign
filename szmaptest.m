%% ============================================================
% Stable monotonic frequency-mapping optimization
%
% Design Omega(w) directly via:
%   dOmega/dw = exp(Q(w)) > 0
%
% Then:
%   Omega(w) = integral_0^w exp(Q(t)) dt
%
% Normalize to enforce:
%   Omega(0) = 0
%   Omega(pi) = pi/T
%
% This guarantees:
%   - strictly monotonic mapping
%   - no folding
%   - no collapse to zero line
%
% Validation:
%   Compare analog prototype Hs(j*2*pi*f)
%   with digital-induced response Hs(j*Omega(w))
%
% NOTE:
%   This step designs the target frequency mapping first.
%   It does NOT yet produce a closed-form rational G(z).
%% ============================================================

clear; clc; close all;

%% ---------------- User parameters ----------------
fs   = 48000;
T    = 1/fs;

fmin = 20;
fmax = fs/2;
Nw   = 4000;

% Q(w) basis order:
K = 8;   % increase to 10 or 12 if needed

% Optimization options
maxIter    = 12000;
displayOpt = 'iter';   % 'off' or 'iter'

% Weights
lambda_hi      = 8;    % high-frequency emphasis
pow_hi         = 3;
lambda_anchor  = 200;
lambda_smooth  = 1e-2;
lambda_qenergy = 1e-3;

eps_log = 1e-12;

% Prototype for validation only
f0 = 22000;
Qp = 5.707;
w0 = 2*pi*f0;

%% ---------------- Frequency grid ----------------
f = logspace(log10(fmin), log10(fmax), Nw);
w = 2*pi*f/fs;                 % in (0, pi]
Omega_target = w / T;

W = 1 + lambda_hi*(w/pi).^pow_hi;

%% ---------------- Anchor points ----------------
f_anchor = [200, 1000, 3000, 8000, 12000, 18000, 22000];
f_anchor = f_anchor(f_anchor < fs/2);
w_anchor = 2*pi*f_anchor/fs;
Omega_anchor_target = w_anchor / T;

%% ---------------- Initial guess ----------------
% q = [q0 ... qK], Q(w)=sum qk*cos(k*w)
q0 = zeros(K+1,1);

%% ---------------- Optimization ----------------
objfun = @(q) objective_monotonic_mapping( ...
    q, w, Omega_target, W, ...
    w_anchor, Omega_anchor_target, ...
    T, lambda_anchor, lambda_smooth, lambda_qenergy, eps_log);

opts = optimset('Display', displayOpt, ...
                'MaxIter', maxIter, ...
                'MaxFunEvals', 200000, ...
                'TolX', 1e-10, ...
                'TolFun', 1e-10);

q_opt = fminsearch(objfun, q0, opts);

%% ---------------- Evaluate optimized mapping ----------------
[Qw, dOmega_raw, Omega_opt] = build_monotonic_mapping(q_opt, w, T);
Omega_blt   = (2/T) * tan(w/2);
Omega_ideal = w / T;

dOmega_opt = numerical_derivative(w, Omega_opt);

fprintf('\nOptimized q coefficients:\n');
disp(q_opt(:).');

fprintf('Min dOmega/dw    = %.12g\n', min(dOmega_opt));
fprintf('Omega(0+)        = %.12g rad/s\n', Omega_opt(1));
fprintf('Omega(pi)        = %.12g rad/s\n', Omega_opt(end));
fprintf('Target pi/T      = %.12g rad/s\n', pi/T);

%% ---------------- Anchor evaluation ----------------
Omega_anchor_opt = interp1(w, Omega_opt, w_anchor, 'pchip');

disp(' ');
disp('Anchor mapping check [Hz]:');
disp(table(f_anchor(:), ...
           (Omega_anchor_target(:)/(2*pi)), ...
           (Omega_anchor_opt(:)/(2*pi)), ...
           'VariableNames', {'f_anchor_Hz','TargetAnalogHz','MappedAnalogHz'}));

%% ---------------- Validation on analog prototype ----------------
Hs     = analog_lp_response(2*pi*f, w0, Qp);
Hz_blt = analog_lp_response(Omega_blt, w0, Qp);
Hz_opt = analog_lp_response(Omega_opt, w0, Qp);

Hs_dB     = 20*log10(abs(Hs) + 1e-15);
Hz_blt_dB = 20*log10(abs(Hz_blt) + 1e-15);
Hz_opt_dB = 20*log10(abs(Hz_opt) + 1e-15);

err_blt = Hz_blt_dB - Hs_dB;
err_opt = Hz_opt_dB - Hs_dB;

fprintf('BLT dB RMSE = %.6f dB\n', rms(err_blt));
fprintf('OPT dB RMSE = %.6f dB\n', rms(err_opt));

%% ---------------- Plots ----------------
figure('Color','w');
semilogx(f, Omega_ideal/(2*pi), 'LineWidth', 1.8); hold on;
semilogx(f, Omega_blt/(2*pi), '--', 'LineWidth', 1.5);
semilogx(f, Omega_opt/(2*pi), 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Digital frequency (Hz)');
ylabel('Mapped analog frequency (Hz)');
title('Frequency mapping: ideal vs BLT vs optimized monotonic mapping');
legend('Ideal \Omega=\omega/T', 'BLT', 'Optimized monotonic', 'Location', 'best');

figure('Color','w');
semilogx(f, log(Omega_blt + eps_log) - log(Omega_ideal + eps_log), '--', 'LineWidth', 1.5); hold on;
semilogx(f, log(Omega_opt + eps_log) - log(Omega_ideal + eps_log), 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('log-mapping error');
title('Log frequency mapping error');
legend('BLT', 'Optimized monotonic', 'Location', 'best');

figure('Color','w');
semilogx(f, Hs_dB, 'LineWidth', 1.8); hold on;
semilogx(f, Hz_blt_dB, '--', 'LineWidth', 1.5);
semilogx(f, Hz_opt_dB, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
ylim([-40 20]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(sprintf('Analog vs BLT vs optimized monotonic mapping | f_s=%g Hz, f_0=%g Hz, Q=%g', fs, f0, Qp));
legend('Analog prototype', 'BLT', 'Optimized monotonic', 'Location', 'best');

figure('Color','w');
semilogx(f, err_blt, '--', 'LineWidth', 1.5); hold on;
semilogx(f, err_opt, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('Magnitude error (dB)');
title('Digital minus Analog magnitude error');
legend('BLT error', 'Optimized monotonic error', 'Location', 'best');

figure('Color','w');
semilogx(f, dOmega_opt, 'LineWidth', 1.6); hold on;
yline(0, '--');
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('d\Omega/d\omega');
title('Monotonicity check (should stay > 0)');

figure('Color','w');
plot(w, Qw, 'LineWidth', 1.5); grid on;
xlabel('\omega (rad/sample)');
ylabel('Q(\omega)');
title('Exponent basis Q(\omega)');

%% ============================================================
%% Local functions
%% ============================================================

function J = objective_monotonic_mapping( ...
    q, w, Omega_target, W, ...
    w_anchor, Omega_anchor_target, ...
    T, lambda_anchor, lambda_smooth, lambda_qenergy, eps_log)

    [Qw, dOmega_raw, Omega] = build_monotonic_mapping(q, w, T);

    % Main fit: log-frequency error
    Elog = log(Omega + eps_log) - log(Omega_target + eps_log);
    Jfit = mean(W(:) .* (Elog(:).^2));

    % Anchor fit
    Omega_anchor = interp1(w, Omega, w_anchor, 'pchip');
    Eanchor = log(Omega_anchor + eps_log) - log(Omega_anchor_target + eps_log);
    Janchor = mean(Eanchor.^2);

    % Smoothness on derivative shape
    d2 = numerical_derivative(w, dOmega_raw);
    Jsmooth = mean(d2.^2);

    % Keep Q modest to avoid pathological overfitting
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
    % Q(w) = q0 + q1*cos(w) + q2*cos(2w) + ...
    Qw = zeros(size(w));
    for k = 0:length(q)-1
        Qw = Qw + q(k+1) * cos(k*w);
    end

    % Positive derivative by construction
    dOmega_raw = exp(Qw);

    % Integrate from 0 to w
    Omega_raw = cumtrapz(w, dOmega_raw);

    % Normalize endpoints:
    % enforce Omega(pi)=pi/T
    scale = (pi/T) / Omega_raw(end);
    Omega = scale * Omega_raw;
end

function H = analog_lp_response(Omega, w0, Q)
    s = 1j * Omega;
    H = (w0^2) ./ (s.^2 + (w0/Q)*s + w0^2);
end

function d = numerical_derivative(x, y)
    d = zeros(size(y));
    d(2:end-1) = (y(3:end) - y(1:end-2)) ./ (x(3:end) - x(1:end-2));
    d(1) = (y(2) - y(1)) / (x(2) - x(1));
    d(end) = (y(end) - y(end-1)) / (x(end) - x(end-1));
end