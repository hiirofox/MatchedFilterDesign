clc; clear; close all;

%% =========================================================
% Parameters
%% =========================================================
fs = 48000;              % Sampling rate [Hz], Nyquist = 24000 Hz
fc = 100;              % Cutoff frequency [Hz]
Q  = 5;                  % Q factor of the original 2nd-order LPF

% Negative a values: emphasize moving low-frequency structure upward
a_list = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1];

% Frequency axis for plotting: logarithmic sampling
f = logspace(log10(20), log10(fs/2), 4000);   % Hz, up to Nyquist = 24000
omega = 2*pi*f/fs;                            % digital rad/sample

%% =========================================================
% 2nd-order digital low-pass design (RBJ cookbook form)
%% =========================================================
w0 = 2*pi*fc/fs;
alpha = sin(w0)/(2*Q);

b0 = (1 - cos(w0))/2;
b1 = 1 - cos(w0);
b2 = (1 - cos(w0))/2;
a0 = 1 + alpha;
a1 = -2*cos(w0);
a2 = 1 - alpha;

b = [b0 b1 b2] / a0;
a = [1 a1/a0 a2/a0];

%% =========================================================
% Evaluate original filter on arbitrary omega
%% =========================================================
ejw  = exp(-1j*omega);
H0 = (b(1) + b(2)*ejw + b(3)*ejw.^2) ./ (1 + a(2)*ejw + a(3)*ejw.^2);
H0_dB = 20*log10(abs(H0) + 1e-12);

%% =========================================================
% Plot
%% =========================================================
figure('Color','w','Position',[100 100 950 600]);
hold on; grid on; box on;

% Original filter in black
semilogx(f, H0_dB, 'k', 'LineWidth', 2.4, 'DisplayName', 'Original');

% Warped filters
cmap = lines(length(a_list));

for k = 1:length(a_list)
    a_apf = a_list(k);

    % Frequency warping:
    % omega' = 2*atan(((1+a)/(1-a))*tan(omega/2))
    omega_p = 2*atan(((1 + a_apf)/(1 - a_apf)) * tan(omega/2));

    % Numerical safety: map to principal range [0, pi]
    omega_p = real(omega_p);
    omega_p(omega_p < 0) = omega_p(omega_p < 0) + pi;

    ejwp = exp(-1j*omega_p);
    Hw = (b(1) + b(2)*ejwp + b(3)*ejwp.^2) ./ (1 + a(2)*ejwp + a(3)*ejwp.^2);
    Hw_dB = 20*log10(abs(Hw) + 1e-12);

    semilogx(f, Hw_dB, 'LineWidth', 1.8, ...
        'Color', cmap(k,:), ...
        'DisplayName', sprintf('a = %.2f', a_apf));
end

%% =========================================================
% Axis formatting
%% =========================================================
set(gca,'XScale','log')
xlim([20 fs/2]);
ylim([-20 20]);

xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title('First-Order All-Pass Frequency Warping of a 2nd-Order Low-Pass Filter', ...
      'FontSize', 13);

% Mark original cutoff
xline(fc, '--k', 'LineWidth', 1.0, ...
    'Label', sprintf('%d Hz', fc), ...
    'LabelVerticalAlignment', 'middle', ...
    'LabelHorizontalAlignment', 'left');

legend('Location', 'southwest');
hold off;