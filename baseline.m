clc
clear
close all

% --- Simulation Parameters ---
X = 50;
Y = 50;
std_mag_noise = 2.0; % Magnetic measurement noise
sigma_step_process = 0.1; % Process noise (real world fluctuation)
sigma_step_measure = 0.3; % Sensor noise (accelerometer error)
sigma_angle_process = 0.05; 
sigma_angle_measure = 0.1;  % Gyro error

% --- Generate Map ---
baseMap = Geometric_Map_Generator(2, [X, Y]);
[xi, yi] = meshgrid(linspace(1, X, 200), linspace(1, Y, 200));
[xx, yy] = meshgrid(1:X, 1:Y);
baseMap_smooth = interp2(xx, yy, baseMap, xi, yi, 'cubic');

figure(1)
imagesc(baseMap_smooth)
colormap(parula);
shading interp;
colorbar
set(gca, 'GridAlpha', 0, 'Box', 'on', 'YDir', 'normal');
title('FGLC Navigation Simulation');
hold on;

% --- Initialization ---
init_pos = [25, 25]; 
true_pos = init_pos;
true_heading = 0;

true_path = true_pos;
pdr_steps = [];
pdr_dangles = [];
mag_measurements = [];

% --- Main Simulation Loop ---
total_steps = 30;
hWait = waitbar(0, 'Simulating...');

for i = 1:total_steps
    % 1. Walker Intention (Random Walk)
    % The person "wants" to walk 1.5m and turn slightly
    cmd_step = 1.5; 
    cmd_turn = 0.2 * randn(); 
    
    % 2. Ground Truth Generation (Physics)
    % Real movement has process noise (e.g., slip, uneven floor)
    actual_step = cmd_step + sigma_step_process * randn();
    actual_turn = cmd_turn + sigma_angle_process * randn();
    
    % Calculate proposed new position
    new_heading = true_heading + actual_turn;
    new_pos = true_pos + actual_step * [cos(new_heading), sin(new_heading)];
    
    % --- WALL BOUNCE LOGIC (The physical reality) ---
    hit_wall = false;
    if new_pos(1) < 2 || new_pos(1) > X-2 || new_pos(2) < 2 || new_pos(2) > Y-2
        hit_wall = true;
        % Physical reaction: Turn around (approx 180 deg)
        actual_turn = actual_turn + pi + 0.1*randn(); 
        new_heading = true_heading + actual_turn;
        % Recalculate position with new heading
        new_pos = true_pos + actual_step * [cos(new_heading), sin(new_heading)];
    end
    
    % Update Truth
    true_pos = new_pos;
    true_heading = new_heading;
    true_path = [true_path; true_pos];
    
    % 3. Sensor Measurement (PDR)
    % The sensor measures what ACTUALLY happened (including the wall bounce!)
    % Measured = Actual + Sensor Noise
    meas_step = actual_step + sigma_step_measure * randn();
    meas_turn = actual_turn + sigma_angle_measure * randn();
    
    pdr_steps = [pdr_steps; meas_step];
    pdr_dangles = [pdr_dangles; meas_turn];
    
    % 4. Magnetic Measurement
    grid_x = linspace(1, X, size(baseMap_smooth, 2));
    grid_y = linspace(1, Y, size(baseMap_smooth, 1));
    true_mag = interp2(grid_x, grid_y, baseMap_smooth, true_pos(1), true_pos(2), 'cubic');
    meas_mag = true_mag + std_mag_noise * randn();
    mag_measurements = [mag_measurements; meas_mag];
    
    % 5. Run FGLC Optimizer
    [est_traj, pdr_traj] = mainalgr(init_pos, pdr_steps, pdr_dangles, mag_measurements, baseMap_smooth, X, Y);

    waitbar(i/total_steps, hWait);
end
close(hWait);

% --- Plotting ---
plot(true_path(:,1), true_path(:,2), 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(pdr_traj(:,1), pdr_traj(:,2), 'g--', 'LineWidth', 1.5, 'DisplayName', 'PDR Only');
plot(est_traj(:,1), est_traj(:,2), 'r--', 'LineWidth', 2, 'DisplayName', 'FGLC (Adam)');
legend show;

%% --- FGLC ALGORITHM (Adam) ---
function [est_traj, pdr_traj] = mainalgr(start_pos, steps, angles, mags, map, maxX, maxY)
    N = length(steps);
    
    % Constraints
    delta_step = 3 * 0.3; 
    delta_theta = 3 * 0.1; 
    
    % Optimizer settings
    lr = 0.05;
    iter_max = 80;
    
    w = zeros(N, 1);
    v = zeros(N, 1);
    
    m_w = zeros(N,1); v_w = zeros(N,1);
    m_v = zeros(N,1); v_v = zeros(N,1);
    
    % PDR Baseline
    pdr_traj = reconstruct_path(start_pos, steps, angles);
    
    grid_x = linspace(1, maxX, size(map, 2));
    grid_y = linspace(1, maxY, size(map, 1));
    
    for iter = 1:iter_max
        % Decode variables
        step_adj = (2/pi) * delta_step * atan(w);
        theta_adj = (2/pi) * delta_theta * atan(v);
        
        curr_steps = steps + step_adj;
        curr_angles = angles + theta_adj;
        
        % Reconstruct
        path = zeros(N+1, 2);
        path(1,:) = start_pos;
        acc_theta = cumsum(curr_angles);
        
        path(2:end, 1) = start_pos(1) + cumsum(curr_steps .* cos(acc_theta));
        path(2:end, 2) = start_pos(2) + cumsum(curr_steps .* sin(acc_theta));
        
        % Gradients
        grad_w = zeros(N, 1);
        grad_v = zeros(N, 1);
        
        dw_const = (2/pi) * delta_step ./ (1 + w.^2);
        dv_const = (2/pi) * delta_theta ./ (1 + v.^2);
        
        for k = 1:N
            % Precompute motion vector for step k
            vec_x = cos(acc_theta(k));
            vec_y = sin(acc_theta(k));
            
            gw_acc = 0;
            gv_acc = 0;
            
            % Accumulate gradients from future points j >= k
            for j = k:N
                px = path(j+1, 1);
                py = path(j+1, 2);
                
                % Boundary check
                if px < 1 || px > maxX || py < 1 || py > maxY
                    dmag_dx = 0; dmag_dy = 0; mag_val = 0;
                else
                    mag_val = interp2(grid_x, grid_y, map, px, py, 'linear');
                    % Gradients
                    d = 0.5;
                    mx_p = interp2(grid_x, grid_y, map, px+d, py, 'linear');
                    mx_m = interp2(grid_x, grid_y, map, px-d, py, 'linear');
                    dmag_dx = (mx_p - mx_m)/(2*d);
                    
                    my_p = interp2(grid_x, grid_y, map, px, py+d, 'linear');
                    my_m = interp2(grid_x, grid_y, map, px, py-d, 'linear');
                    dmag_dy = (my_p - my_m)/(2*d);
                end
                
                residual = mag_val - mags(j);
                loss_grad = 2 * residual; % Simplified loss
                
                % d(Step)/dw contribution
                gw_acc = gw_acc + loss_grad * (dmag_dx * vec_x + dmag_dy * vec_y);
                
                % d(Heading)/dv contribution (Lever Arm)
                rx = px - path(k, 1); % Vector from start of turn k to point j
                ry = py - path(k, 2);
                
                % Rotation derivative: [-ry, rx]
                gv_acc = gv_acc + loss_grad * (dmag_dx * (-ry) + dmag_dy * (rx));
            end
            
            grad_w(k) = gw_acc * dw_const(k) + 0.1 * w(k); % Add small regularization
            grad_v(k) = gv_acc * dv_const(k) + 0.1 * v(k);
        end
        
        % Adam Update
        beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
        m_w = beta1*m_w + (1-beta1)*grad_w;
        v_w = beta2*v_w + (1-beta2)*grad_w.^2;
        w = w - lr * m_w ./ (sqrt(v_w) + eps);
        
        m_v = beta1*m_v + (1-beta1)*grad_v;
        v_v = beta2*v_v + (1-beta2)*grad_v.^2;
        v = v - lr * m_v ./ (sqrt(v_v) + eps);
    end
    est_traj = path;
end

function traj = reconstruct_path(start, steps, angles)
    traj = zeros(length(steps)+1, 2);
    traj(1,:) = start;
    theta = cumsum(angles);
    traj(2:end,1) = start(1) + cumsum(steps.*cos(theta));
    traj(2:end,2) = start(2) + cumsum(steps.*sin(theta));
end

function map = Geometric_Map_Generator(type, size_arr)
    [xx, yy] = meshgrid(1:size_arr(1), 1:size_arr(2));
    map = 40 + 30 * peaks(size_arr(1)); 
end