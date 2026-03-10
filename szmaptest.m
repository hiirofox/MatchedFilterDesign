%% ============================================================
% Direct G(z) optimization without BLT shell
%
% G(z) = j*(2/T)*R(c),   c = (z + z^-1)/2
%
% On unit circle z = e^{jw}:
%   c = cos(w) in R
%   G(e^{jw}) = j*Omega(w)
%   Omega(w) = (2/T)*R(cos(w))
%
% We optimize R(c) directly to approximate:
%   Omega_target(w) = w/T
%
% Validation is done on a 2nd-order analog lowpass prototype:
%   Hs(s) = w0^2 / (s^2 + (w0/Q)s + w0^2)
%
% This optimization is prototype-independent.
%% ============================================================

clear; clc; close all;

%% ---------------- User parameters ----------------
fs   = 48000;
T    = 1/fs;

fmin = 20;
fmax = fs/2;
Nw   = 3000;

% Rational model order:
% R(c) = (1-c)*(a0 + a1*c + ... + am*c^m) / (1 + b1*c + ... + bn*c^n)
m = 4;   % numerator polynomial order
n = 4;   % denominator polynomial order

% Optimization
maxIter    = 3000;
displayOpt = 'iter';     % 'off' or 'iter'

% weights / penalties
lambda_hi    = 10;       % emphasize high frequency
pow_hi       = 4;
lambda_mono  = 1e4;      % monotonicity penalty
lambda_pos   = 1e4;      % positivity penalty on Omega
lambda_den   = 1e4;      % denominator positivity penalty
lambda_smooth= 1e1;      % smoothness penalty

% Validation analog prototype
f0 = 8000;
Q  = 5.707;
w0 = 2*pi*f0;

%% ---------------- Frequency grid ----------------
f = logspace(log10(fmin), log10(fmax), Nw);
w = 2*pi*f/fs;                 % 0..pi
c = cos(w);

Omega_target = w / T;          % prototype-independent target

W = 1 + lambda_hi*(w/pi).^pow_hi;

%% ---------------- Initial guess ----------------
% theta = [a0 ... am  b1 ... bn]
theta0 = zeros(1, (m+1) + n);

% rough low-frequency slope initialization:
% near w=0, c ~ 1 - w^2/2, so (1-c) ~ w^2/2
% thus this model naturally starts quadratically near DC, not linearly.
% that's a structural limitation of pure R(cos w) model.
% still, let's initialize mildly.
theta0(1) = 1.0;

%% ---------------- Optimization ----------------
objfun = @(th) objective_direct_G(th, c, w, T, Omega_target, W, m, n, ...
                                  lambda_mono, lambda_pos, lambda_den, lambda_smooth);

opts = optimset('Display', displayOpt, ...
                'MaxIter', maxIter, ...
                'MaxFunEvals', 50000, ...
                'TolX', 1e-9, ...
                'TolFun', 1e-9);

theta_opt = fminsearch(objfun, theta0, opts);

[a, b] = unpack_theta(theta_opt, m, n);

fprintf('\nOptimized coefficients:\n');
fprintf('a = '); disp(a);
fprintf('b = '); disp(b);

%% ---------------- Evaluate mapping ----------------
[R_opt, num_opt, den_opt] = eval_R_model(c, a, b);
Omega_opt = (2/T) * R_opt;

% BLT for reference
Omega_blt = (2/T) * tan(w/2);

% derivatives
dOmega_opt = numerical_derivative(w, Omega_opt);

fprintf('Min Omega_opt = %.6g\n', min(Omega_opt));
fprintf('Min dOmega/dw = %.6g\n', min(dOmega_opt));
fprintf('Omega_opt(end)/(2pi) = %.6f Hz\n', Omega_opt(end)/(2*pi));

%% ---------------- Validate on analog prototype ----------------
Hs = analog_lp_response(2*pi*f, w0, Q);

% Digital response induced by G(e^jw)=j*Omega(w)
Hz_blt = analog_lp_response(Omega_blt, w0, Q);
Hz_opt = analog_lp_response(Omega_opt, w0, Q);

Hs_dB     = 20*log10(abs(Hs) + 1e-15);
Hz_blt_dB = 20*log10(abs(Hz_blt) + 1e-15);
Hz_opt_dB = 20*log10(abs(Hz_opt) + 1e-15);

err_blt = Hz_blt_dB - Hs_dB;
err_opt = Hz_opt_dB - Hs_dB;

fprintf('BLT dB RMSE = %.6f dB\n', rms(err_blt));
fprintf('OPT dB RMSE = %.6f dB\n', rms(err_opt));

%% ---------------- Plots ----------------
figure('Color','w');
semilogx(f, Omega_target/(2*pi), 'LineWidth', 1.8); hold on;
semilogx(f, Omega_blt/(2*pi), '--', 'LineWidth', 1.5);
semilogx(f, Omega_opt/(2*pi), 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Digital frequency (Hz)');
ylabel('Mapped analog frequency (Hz)');
title('Frequency mapping: ideal vs BLT vs direct G(z)');
legend('Ideal \Omega=\omega/T', 'BLT', 'Direct G(z)', 'Location', 'best');

figure('Color','w');
semilogx(f, (Omega_blt - Omega_target)/(2*pi), '--', 'LineWidth', 1.5); hold on;
semilogx(f, (Omega_opt - Omega_target)/(2*pi), 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Digital frequency (Hz)');
ylabel('Mapping error (Hz)');
title('Frequency mapping error');
legend('BLT error', 'Direct G(z) error', 'Location', 'best');

figure('Color','w');
semilogx(f, Hs_dB, 'LineWidth', 1.8); hold on;
semilogx(f, Hz_blt_dB, '--', 'LineWidth', 1.5);
semilogx(f, Hz_opt_dB, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
ylim([-40 10]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(sprintf('Analog vs BLT vs Direct G(z) | f_s=%g Hz, f_0=%g Hz, Q=%g', fs, f0, Q));
legend('Analog prototype', 'BLT', 'Direct G(z)', 'Location', 'best');

figure('Color','w');
semilogx(f, err_blt, '--', 'LineWidth', 1.5); hold on;
semilogx(f, err_opt, 'LineWidth', 1.8);
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('Magnitude error (dB)');
title('Digital minus Analog magnitude error');
legend('BLT error', 'Direct G(z) error', 'Location', 'best');

figure('Color','w');
semilogx(f, dOmega_opt, 'LineWidth', 1.6);
grid on;
xlim([fmin fmax]);
xlabel('Frequency (Hz)');
ylabel('d\Omega/d\omega');
title('Monotonicity check of direct G(z) mapping');

%% ---------------- Print final G(z) form ----------------
disp(' ');
disp('Final mapping form:');
disp('G(z) = j*(2/T)*R(c),  c=(z+z^-1)/2');
disp('R(c) = (1-c)*(a0 + a1*c + ... + am*c^m) / (1 + b1*c + ... + bn*c^n)');

%% ============================================================
%% Local functions
%% ============================================================

function J = objective_direct_G(theta, c, w, T, Omega_target, W, m, n, ...
                                lambda_mono, lambda_pos, lambda_den, lambda_smooth)

    [a, b] = unpack_theta(theta, m, n);
    [R, ~, den] = eval_R_model(c, a, b);

    Omega = (2/T) * R;

    % fit mapping directly
    E = Omega - Omega_target;
    Jfit = mean(W(:) .* (E(:).^2));

    % positivity of mapping
    pos_violation = max(0, -Omega + 1e-8);
    Jpos = mean(pos_violation.^2);

    % monotonicity
    dOmega = numerical_derivative(w, Omega);
    mono_violation = max(0, -dOmega + 1e-8);
    Jmono = mean(mono_violation.^2);

    % denominator should not cross zero on the fitting interval
    den_violation = max(0, -den + 1e-6);
    Jden = mean(den_violation.^2);

    % smoothness / avoid pathological oscillation
    d2Omega = numerical_derivative(w, dOmega);
    Jsmooth = mean(d2Omega.^2);

    J = Jfit ...
      + lambda_pos   * Jpos ...
      + lambda_mono  * Jmono ...
      + lambda_den   * Jden ...
      + lambda_smooth* Jsmooth;

    if any(~isfinite(R)) || any(~isfinite(Omega)) || ~isfinite(J)
        J = 1e30;
    end
end

function [a, b] = unpack_theta(theta, m, n)
    a = theta(1:m+1);
    b = theta(m+2:m+1+n);
end

function [R, num, den] = eval_R_model(c, a, b)
    % numpoly = a0 + a1*c + ... + am*c^m
    numpoly = zeros(size(c));
    for k = 0:length(a)-1
        numpoly = numpoly + a(k+1) * c.^k;
    end

    % den = 1 + b1*c + ... + bn*c^n
    den = ones(size(c));
    for k = 1:length(b)
        den = den + b(k) * c.^k;
    end

    num = (1 - c) .* numpoly;
    R = num ./ den;
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