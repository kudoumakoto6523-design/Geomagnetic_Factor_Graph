clc; clear; close all;

%% =================== 0) GPU CHECK ===================
if gpuDeviceCount == 0
    error('No GPU detected. Need Parallel Computing Toolbox.');
end
d = gpuDevice;
fprintf('Running on GPU: %s\n', d.Name);

%% =================== 1) Load REAL MAG only (no steps/angles from file) ===================
S1 = load('test1_inputs.mat');
S2 = load('test2_inputs.mat');

mags1 = double(S1.mag_measurements(:));   % REAL MAG
mags2 = double(S2.mag_measurements(:));   % REAL MAG

N1 = numel(mags1);
N2 = numel(mags2);

%% =================== 2) Your TRUE routes (you told me) ===================
% test1: (0,1)->(5,1)
GT1.start = [0, 1];
GT1.end   = [5, 1];

% test2: (4,1)->(4,7)
GT2.start = [4, 1];
GT2.end   = [4, 7];

%% =================== 3) Build "truth steps/angles" from the real path ===================
% We make N steps between start and end (so there are N+1 points).
[truth1, steps_true1, theta0_1] = build_truth_from_line(GT1.start, GT1.end, N1);
[truth2, steps_true2, theta0_2] = build_truth_from_line(GT2.start, GT2.end, N2);

% truth dangles is 0 (straight line). We'll put the initial heading in theta0.
angles_true1 = zeros(N1,1);
angles_true2 = zeros(N2,1);

%% =================== 4) Make a "drifting PDR" from truth (simple controlled drift) ===================
% This is NOT noise. It's a deterministic drift per step (like gyro bias).
drift_per_step_deg = 0.50;                  % <<< ONLY CHANGE THIS LINE
drift_per_step = drift_per_step_deg*pi/180;

% PDR angle increments = truth increments + constant drift increment
angles_pdr1 = angles_true1 + drift_per_step;
angles_pdr2 = angles_true2 + drift_per_step;

% (Optional) step scale error if you want: e.g., 1.02
step_scale = 1.00;                         % <<< keep 1.00 if you want "only heading drift"
steps_pdr1 = steps_true1 * step_scale;
steps_pdr2 = steps_true2 * step_scale;

% Build the drifted PDR trajectories just for plotting
pdr1 = integrate_traj(GT1.start, steps_pdr1, angles_pdr1, theta0_1);
pdr2 = integrate_traj(GT2.start, steps_pdr2, angles_pdr2, theta0_2);

%% =================== 5) Kriging map from your 7x5 tile ===================
tile_mag_7x5 = [ ...
    0.0400, 0.0330, 0.0250, 0.0245, 0.0450; ...
    0.0360, 0.0300, 0.0280, 0.0300, 0.0210; ...
    0.0280, 0.0330, 0.0460, 0.0540, 0.0530; ...
    0.0530, 0.0406, 0.0520, 0.0524, 0.0428; ...
    0.0510, 0.0490, 0.0400, 0.0345, 0.0536; ...
    0.0445, 0.0435, 0.0488, 0.0508, 0.0720; ...
    0.0500, 0.0620, 0.0670, 0.0430, 0.0770  ...
];

if exist('fitrgp','file') ~= 2
    error('Need fitrgp for Kriging (Statistics and Machine Learning Toolbox).');
end

% Coordinates: x in [0..5] with 5 samples, y in [0..7] with 7 samples
x_nodes = linspace(0,5,5);
y_nodes = linspace(0,7,7);

[Xn, Yn] = meshgrid(x_nodes, y_nodes);
XY = [Xn(:), Yn(:)];
Z  = tile_mag_7x5(:);

gprMdl = fitrgp(XY, Z, ...
    'Basis','constant', ...
    'KernelFunction','ardsquaredexponential', ...
    'Sigma', 1e-6, ...
    'FitMethod','exact', ...
    'PredictMethod','exact', ...
    'Standardize', true);

% Dense grid
grid_x = linspace(min(x_nodes), max(x_nodes), 401);
grid_y = linspace(min(y_nodes), max(y_nodes), 561);
[XXq, YYq] = meshgrid(grid_x, grid_y);
Zq = predict(gprMdl, [XXq(:), YYq(:)]);
magMapDense = reshape(Zq, size(XXq));

% Move to GPU once
g_map    = gpuArray(single(magMapDense));
g_grid_x = gpuArray(single(grid_x));
g_grid_y = gpuArray(single(grid_y));

%% =================== 6) FGLC parameters (NOT adding noise; just weights/limits) ===================
sigma_mag    = 0.003;   % smaller => mags pull harder
sigma_step   = 0.010;    % step correction allowance scale
sigma_dangle = 0.15;    % heading correction allowance scale

lr = 0.05;
iter_max = 500;

%% =================== 7) Run FGLC (one shot) ===================
est1_gpu = mainalgr_gpu( GT1.start, steps_pdr1, angles_pdr1, mags1, ...
    g_map, g_grid_x, g_grid_y, theta0_1, sigma_mag, sigma_step, sigma_dangle, lr, iter_max);

est2_gpu = mainalgr_gpu( GT2.start, steps_pdr2, angles_pdr2, mags2, ...
    g_map, g_grid_x, g_grid_y, theta0_2, sigma_mag, sigma_step, sigma_dangle, lr, iter_max);

est1 = gather(est1_gpu);
est2 = gather(est2_gpu);

%% =================== 8) Plot ===================
figure('Name','REAL MAG + Drifted PDR + FGLC (Kriging map)');
imagesc(grid_x, grid_y, magMapDense);
set(gca,'YDir','normal'); axis equal tight;
colormap(parula); colorbar; hold on;
title(sprintf('REAL MAG, Drift PDR (%.1f deg/step), FGLC', drift_per_step_deg));
xlabel('X'); ylabel('Y');

plot(truth1(:,1), truth1(:,2), 'k-', 'LineWidth', 2, 'DisplayName','Truth (test1)');
plot(pdr1(:,1),   pdr1(:,2),   'g--','LineWidth', 1.5,'DisplayName','Drift PDR (test1)');
plot(est1(:,1),   est1(:,2),   'r.-','LineWidth', 2, 'DisplayName','FGLC (test1)');

plot(truth2(:,1), truth2(:,2), 'k-', 'LineWidth', 2, 'DisplayName','Truth (test2)');
plot(pdr2(:,1),   pdr2(:,2),   'g--','LineWidth', 1.5,'DisplayName','Drift PDR (test2)');
plot(est2(:,1),   est2(:,2),   'r.-','LineWidth', 2, 'DisplayName','FGLC (test2)');

plot(GT1.start(1), GT1.start(2), 'ko', 'MarkerFaceColor','k', 'DisplayName','Start');
plot(GT2.end(1),   GT2.end(2),   'ks', 'MarkerFaceColor','k', 'DisplayName','End');

legend('Location','bestoutside');
fprintf('Done.\n');

%% =====================================================================
%% ========================= FUNCTIONS =================================
%% =====================================================================

function [truth_traj, steps_true, theta0] = build_truth_from_line(start_xy, end_xy, N)
% Build (N+1)x2 truth line, N steps, constant heading theta0
    truth_traj = [linspace(start_xy(1), end_xy(1), N+1).', ...
                  linspace(start_xy(2), end_xy(2), N+1).'];
    dxy = diff(truth_traj,1,1);
    steps_true = sqrt(sum(dxy.^2,2));
    theta0 = atan2(end_xy(2)-start_xy(2), end_xy(1)-start_xy(1)); % initial heading
end

function traj = integrate_traj(start_xy, steps, dangles, theta0)
% Integrate trajectory from steps + dangles, with initial heading theta0
    theta = theta0 + cumsum(dangles(:));
    px = start_xy(1) + cumsum(steps(:) .* cos(theta));
    py = start_xy(2) + cumsum(steps(:) .* sin(theta));
    traj = [start_xy; [px, py]];
end

function est_traj = mainalgr_gpu(start_pos, steps, angles, mags, g_map, g_x, g_y, ...
                                 theta0, sigma_mag, sigma_step, sigma_dangle, lr, iter_max)
% GPU FGLC: inputs are REAL mags, and your provided steps/angles (here: drifted PDR).
    N = length(steps);

    sigma_mag_g    = gpuArray(single(sigma_mag));
    sigma_step_g   = gpuArray(single(sigma_step));
    sigma_dangle_g = gpuArray(single(sigma_dangle));

    delta_step   = gpuArray(single(3*sigma_step));
    delta_dangle = gpuArray(single(3*sigma_dangle));

    steps_g  = gpuArray(single(steps));
    angles_g = gpuArray(single(angles));
    mags_g   = gpuArray(single(mags));

    w = zeros(N, 1, 'single', 'gpuArray');
    v = zeros(N, 1, 'single', 'gpuArray');

    m_w = zeros(N,1,'single','gpuArray'); vv_w = m_w;
    m_v = zeros(N,1,'single','gpuArray'); vv_v = m_v;
    beta1 = 0.9; beta2 = 0.999; eps_val = 1e-8;

    sp_x = gpuArray(single(start_pos(1)));
    sp_y = gpuArray(single(start_pos(2)));
    theta0_g = gpuArray(single(theta0));

    for iter = 1:iter_max
        step_adj  = (2/pi) * delta_step   * atan(w);
        theta_adj = (2/pi) * delta_dangle * atan(v);

        curr_steps = steps_g + step_adj;
        curr_theta = theta0_g + cumsum(angles_g + theta_adj);

        px = sp_x + cumsum(curr_steps .* cos(curr_theta));
        py = sp_y + cumsum(curr_steps .* sin(curr_theta));

        % map gradients (finite diff)
        d = 0.05;
        qx = px(:); qy = py(:);

        mag_vals = interp2(g_x, g_y, g_map, qx, qy, 'linear', 0);

        mx_p = interp2(g_x, g_y, g_map, qx+d, qy, 'linear', 0);
        mx_m = interp2(g_x, g_y, g_map, qx-d, qy, 'linear', 0);
        dm_dx = (mx_p - mx_m) / (2*d);

        my_p = interp2(g_x, g_y, g_map, qx, qy+d, 'linear', 0);
        my_m = interp2(g_x, g_y, g_map, qx, qy-d, 'linear', 0);
        dm_dy = (my_p - my_m) / (2*d);

        residual = (2/(sigma_mag_g^2)) * (mag_vals - mags_g);
        term_x = residual .* dm_dx;
        term_y = residual .* dm_dy;

        grad_w = zeros(N,1,'single','gpuArray'); % dJ/d(step_adj) then chain to w
        grad_v = zeros(N,1,'single','gpuArray'); % dJ/d(theta_adj) then chain to v

        for k = 1:N
            grad_w(k) = sum(term_x(k:end)) * cos(curr_theta(k)) + ...
                        sum(term_y(k:end)) * sin(curr_theta(k));

            if k == 1
                pk_x = sp_x; pk_y = sp_y;
            else
                pk_x = px(k-1); pk_y = py(k-1);
            end
            rx = px(k:end) - pk_x;
            ry = py(k:end) - pk_y;
            grad_v(k) = sum(term_x(k:end) .* (-ry) + term_y(k:end) .* (rx));
        end

        % add step/dangle penalties (paper’s missing terms)
        grad_w = grad_w + (2/(sigma_step_g^2))   * step_adj;
        grad_v = grad_v + (2/(sigma_dangle_g^2)) * theta_adj;

        % chain rule for atan mapping
        dw_const = (2/pi) * delta_step   ./ (1 + w.^2);
        dv_const = (2/pi) * delta_dangle ./ (1 + v.^2);
        grad_w = grad_w .* dw_const;
        grad_v = grad_v .* dv_const;

        % Adam
        m_w  = beta1*m_w  + (1-beta1)*grad_w;
        vv_w = beta2*vv_w + (1-beta2)*(grad_w.^2);
        w = w - lr * m_w ./ (sqrt(vv_w) + eps_val);

        m_v  = beta1*m_v  + (1-beta1)*grad_v;
        vv_v = beta2*vv_v + (1-beta2)*(grad_v.^2);
        v = v - lr * m_v ./ (sqrt(vv_v) + eps_val);
    end

    % output final trajectory
    step_adj  = (2/pi) * delta_step   * atan(w);
    theta_adj = (2/pi) * delta_dangle * atan(v);

    curr_steps = steps_g + step_adj;
    curr_theta = theta0_g + cumsum(angles_g + theta_adj);

    px = sp_x + cumsum(curr_steps .* cos(curr_theta));
    py = sp_y + cumsum(curr_steps .* sin(curr_theta));

    est_traj = [ [sp_x, sp_y]; [px, py] ];
end
