%la prima funzione è nel caso nomainale, la seconda commentata è per
%gestire la robustezza.

clearvars; close all; clc;

% Configuration
config = struct();

% Set random seed for reproducibility
config.seed = 42;
rng(config.seed);

% Equilibrium points
config.xA = [7.87298; 0.12702];  % Stable node (avoid)
config.xB = [0.12702; 7.87298];  % Stable node (start)
config.xS = [1.92335; 1.92335];  % Saddle point (target)

% Environment parameters
config.xMax = 10.0;
config.dt = 0.05;
config.u_min = -10;
config.u_max = 30;

% Episode settings
config.Nsteps = 1000;  % Max steps per episode (50 seconds)
config.saddleRadius = 0.20;
config.holdSteps = 40;  % Hold for 2 seconds (40 stepsx0.05s)
config.successBonus = 50;

% State space discretization
config.fineHalfWidth = 3.0;  % High resolution region around saddle
config.stepFine = 0.5;      
config.stepCoarse = 1.25;   

% Action space discretization
config.nActions = 8;

% Learning hyperparameters
config.Nep = 6000;           % Number of episodes
config.gamma = 0.90;         % Discount factor
config.alpha = 0.80;         % Learning rate

% Exploration parameters
config.epsilon0 = 0.20;      % Initial exploration rate
config.epsilonMin = 0.05;    % Minimum exploration rate
config.epsDecay = 0.9995;    % Decay rate per episode

% Curriculum learning
config.Ncurr = 3500;         % Episodes for curriculum phase

% Early stopping
config.earlyStopWindow = 500;    % Window for moving average
config.earlyStopThreshold = 30;  % Convergence threshold


config.saveResults = true;
config.savePath = fullfile(fileparts(mfilename('fullpath')), 'outputs');
config.checkpoint_interval = 1000; % Save every N episodes

% Initialize environment & Create Output Folder
fprintf('=== Tabular Q-Learning for Toggle Switch ===\n');

% FIX: Create directory if it doesn't exist
if config.saveResults 
    if ~isfolder(config.savePath)
        try
            mkdir(config.savePath);
            fprintf('Created output directory: %s\n', config.savePath);
        catch
            fprintf('WARNING: Could not create output directory. Saving disabled.\n');
            config.saveResults = false;
        end
    end
end

fprintf('Configuration:\n');
fprintf('  Episodes: %d\n', config.Nep);
fprintf('  Curriculum episodes: %d\n', config.Ncurr);
fprintf('  Learning rate: %.2f\n', config.alpha);
fprintf('  Discount factor: %.2f\n', config.gamma);
fprintf('  Random seed: %d\n\n', config.seed);

% Create non-uniform discretization
[bin_x1, bin_x2] = createNonUniformBins(config);
config.bin_x1 = bin_x1;
config.bin_x2 = bin_x2;
config.nStates = [numel(bin_x1), numel(bin_x2)];

% Create action bins
config.bin_u = linspace(config.u_min, config.u_max, config.nActions);

fprintf('State space: %d × %d = %d states\n', ...
    config.nStates(1), config.nStates(2), prod(config.nStates));
fprintf('Action space: %d discrete actions\n\n', config.nActions);

% Initialize Q-table
Qtable = zeros(config.nStates(1), config.nStates(2), config.nActions);

%% Training
fprintf('Starting training...\n');

% Training metrics
scores = zeros(config.Nep, 1);
episodeLengths = zeros(config.Nep, 1);
successFlags = false(config.Nep, 1);
epsilonHistory = zeros(config.Nep, 1);

epsilon = config.epsilon0;
bestScore = -inf;
converged = false;

tic;

for ep = 1:config.Nep
    % Curriculum learning schedule
    if ep <= config.Ncurr
        lambda = ep / config.Ncurr;
    else
        lambda = 1.0;
    end

    % Reset environment
    State = toggleReset(lambda);

    % Episode variables
    score = 0;
    holdCounter = 0;
    episodeSuccess = false;

    % Run episode
    for k = 1:config.Nsteps
        % Discretize state
        stateIdx = discretizeState(State, config);

        % Select action (epsilon-greedy)
        if rand() > epsilon
            % Exploit: choose best action
            [~, actionIdx] = max(Qtable(stateIdx(1), stateIdx(2), :));
        else
            % Explore: random action
            actionIdx = randi(config.nActions);
        end

        action = config.bin_u(actionIdx);

        % Take step in environment
        [NextState, reward, isDone, ~] = toggleStep(action, State);

        % Check for success (holding near saddle)
        distToSaddle = norm(NextState - config.xS, 2);
        if distToSaddle <= config.saddleRadius
            holdCounter = holdCounter + 1;
            if holdCounter >= config.holdSteps
                reward = reward + config.successBonus;
                isDone = true;
                episodeSuccess = true;
            end
        else
            holdCounter = 0;
        end

        % Q-learning update
        nextStateIdx = discretizeState(NextState, config);

        currentQ = Qtable(stateIdx(1), stateIdx(2), actionIdx);
        maxNextQ = max(Qtable(nextStateIdx(1), nextStateIdx(2), :));

        tdTarget = reward + config.gamma * maxNextQ;
        tdError = tdTarget - currentQ;

        Qtable(stateIdx(1), stateIdx(2), actionIdx) = ...
            currentQ + config.alpha * tdError;

        % Update state and score
        State = NextState;
        score = score + reward;

        % Check termination
        if isDone
            break;
        end
    end

    % Store metrics
    scores(ep) = score;
    episodeLengths(ep) = k;
    successFlags(ep) = episodeSuccess;
    epsilonHistory(ep) = epsilon;

    % Update best score
    if score > bestScore
        bestScore = score;
    end

    % Decay epsilon
    epsilon = max(config.epsilonMin, epsilon * config.epsDecay);

    % Progress reporting
    if mod(ep, 100) == 0
        recentSuccess = sum(successFlags(max(1,ep-99):ep));
        avgScore = mean(scores(max(1,ep-99):ep));
        fprintf('Episode %d/%d | Avg Score: %.1f | Success: %d/100 | Epsilon: %.3f\n', ...
            ep, config.Nep, avgScore, recentSuccess, epsilon);
    end

    % Checkpointing (CRASH PROOF)
    if config.saveResults && mod(ep, config.checkpoint_interval) == 0
        try
            filename = fullfile(config.savePath, sprintf('checkpoint_ep%d.mat', ep));
            save(filename, 'Qtable', 'scores', 'config', 'ep');
            fprintf('Saved checkpoint: %s\n', filename);
        catch ME
            fprintf('WARNING: Save failed (Permission/Path error). Continuing training...\n');
        end
    end

    % Early stopping check
    if ep > config.earlyStopWindow
        recentScores = scores(ep-config.earlyStopWindow+1:ep);
        if mean(recentScores) > config.earlyStopThreshold && ...
           std(recentScores) < 5
            fprintf('\nConverged at episode %d!\n', ep);
            converged = true;
            % Trim arrays
            scores = scores(1:ep);
            episodeLengths = episodeLengths(1:ep);
            successFlags = successFlags(1:ep);
            epsilonHistory = epsilonHistory(1:ep);
            break;
        end
    end
end

trainingTime = toc;
fprintf('\nTraining completed in %.1f seconds\n', trainingTime);
fprintf('Best score achieved: %.1f\n', bestScore);
fprintf('Final success rate (last 100 ep): %.1f%%\n', ...
    100 * sum(successFlags(end-99:end)) / 100);

%% Save final results 
if config.saveResults
    try
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        filename = sprintf('qtable_final_%s.mat', timestamp);
        fullPath = fullfile(config.savePath, filename);
        save(fullPath, ...
            'Qtable', 'scores', 'episodeLengths', 'successFlags', ...
            'config', 'trainingTime');
        fprintf('Results saved to: %s\n', fullPath);
    catch ME
        fprintf('ERROR: Could not save final results. Check folder permissions.\n');
    end
end

%Validation (Greedy Policy)
fprintf('\nRunning validation with greedy policy...\n');
State = toggleReset(1.0); % Start from xB

% Preallocate history
stateHistory = zeros(config.Nsteps, 2);
actionHistory = zeros(config.Nsteps, 1);
rewardHistory = zeros(config.Nsteps, 1);

stateHistory(1, :) = State';
score = 0;
holdCounter = 0;

for k = 2:config.Nsteps
    % Greedy action selection
    stateIdx = discretizeState(State, config);
    [~, actionIdx] = max(Qtable(stateIdx(1), stateIdx(2), :));
    action = config.bin_u(actionIdx);

    % Take step
    [NextState, reward, isDone, ~] = toggleStep(action, State);

    % Check success
    distToSaddle = norm(NextState - config.xS, 2);
    if distToSaddle <= config.saddleRadius
        holdCounter = holdCounter + 1;
        if holdCounter >= config.holdSteps
            reward = reward + config.successBonus;
            isDone = true;
        end
    else
        holdCounter = 0;
    end

    % Store history
    stateHistory(k, :) = NextState';
    actionHistory(k) = action;
    rewardHistory(k) = reward;
    score = score + reward;

    State = NextState;

    if isDone
        % Trim arrays
        stateHistory = stateHistory(1:k, :);
        actionHistory = actionHistory(1:k);
        rewardHistory = rewardHistory(1:k);
        break;
    end
end

fprintf('Validation score: %.1f\n', score);
fprintf('Episode length: %d steps (%.1f seconds)\n', k, k * config.dt);

%% Visualization
createTrainingPlots(scores, episodeLengths, successFlags, epsilonHistory, config);
createValidationPlots(stateHistory, actionHistory, rewardHistory, config);

%% Helper Functions

function [bin_x1, bin_x2] = createNonUniformBins(config)
% createNonUniformBins - Create non-uniform discretization with fine grid near saddle
    bin_x1 = nonuniformBins(0, config.xMax, config.xS(1), ...
        config.fineHalfWidth, config.stepFine, config.stepCoarse);
    bin_x2 = nonuniformBins(0, config.xMax, config.xS(2), ...
        config.fineHalfWidth, config.stepFine, config.stepCoarse);
end

function bins = nonuniformBins(xmin, xmax, xref, halfWidth, hFine, hCoarse)
% nonuniformBins - Create non-uniform bins with fine resolution near reference point
    leftEnd = max(xmin, xref - halfWidth);
    rightStart = min(xmax, xref + halfWidth);

    % Create three regions
    bins_left = xmin:hCoarse:leftEnd;
    bins_center = leftEnd:hFine:rightStart;
    bins_right = rightStart:hCoarse:xmax;

    % Combine and remove duplicates
    bins = unique([bins_left, bins_center, bins_right]);

    % Ensure bounds are included
    if bins(end) < xmax
        bins(end+1) = xmax;
    end
    if bins(1) > xmin
        bins = [xmin, bins];
    end
end

function idx = discretizeState(state, config)
% discretizeState - Map continuous state to discrete bin indices
    [~, idx1] = min(abs(state(1) - config.bin_x1));
    [~, idx2] = min(abs(state(2) - config.bin_x2));
    idx = [idx1, idx2];
end

function createTrainingPlots(scores, episodeLengths, successFlags, epsilonHistory, config)
% createTrainingPlots - Create comprehensive training visualization

    figure('Position', [100, 100, 1200, 800], 'Color', 'w');

    % Learning curve
    subplot(2, 2, 1);
    smoothWindow = min(200, floor(length(scores)/10));
    plot(movmean(scores, smoothWindow), 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
    hold on;
    plot(scores, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
    xlabel('Episode');
    ylabel('Cumulative Reward');
    title('Learning Curve');
    legend({'Smoothed', 'Raw'}, 'Location', 'southeast');
    grid on;
    set(gca, 'FontSize', 12);

    % Success rate
    subplot(2, 2, 2);
    windowSize = 100;
    successRate = movmean(double(successFlags), windowSize) * 100;
    plot(successRate, 'LineWidth', 2, 'Color', [0.2, 0.7, 0.3]);
    xlabel('Episode');
    ylabel('Success Rate (%)');
    title(sprintf('Success Rate (Window = %d)', windowSize));
    ylim([0, 105]);
    grid on;
    set(gca, 'FontSize', 12);

    % Episode length
    subplot(2, 2, 3);
    plot(movmean(episodeLengths, smoothWindow), 'LineWidth', 2, 'Color', [0.8, 0.4, 0.2]);
    xlabel('Episode');
    ylabel('Steps');
    title('Episode Length');
    grid on;
    set(gca, 'FontSize', 12);

    % Exploration rate
    subplot(2, 2, 4);
    plot(epsilonHistory, 'LineWidth', 2, 'Color', [0.6, 0.2, 0.7]);
    xlabel('Episode');
    ylabel('Epsilon');
    title('Exploration Rate Decay');
    grid on;
    set(gca, 'FontSize', 12);

    sgtitle('Training Metrics', 'FontSize', 16, 'FontWeight', 'bold');
end

function createValidationPlots(stateHistory, actionHistory, rewardHistory, config)

    Nsim = size(stateHistory, 1);
    t = (0:Nsim-1) * config.dt;

    % Normalized error
    e1n = (stateHistory(:,1) - config.xS(1)) / config.xS(1);
    e2n = (stateHistory(:,2) - config.xS(2)) / config.xS(2);
    errorNorm = sqrt(e1n.^2 + e2n.^2);

    figure('Position', [150, 150, 1400, 900], 'Color', 'w');

    % State trajectories
    subplot(3, 2, 1);
    plot(t, stateHistory(:,1), 'LineWidth', 2, 'DisplayName', '$x_1$');
    hold on;
    plot(t, stateHistory(:,2), 'LineWidth', 2, 'DisplayName', '$x_2$');
    yline(config.xS(1), '--k', 'LineWidth', 1.5, 'DisplayName', '$x_S$');
    xlabel('Time [min]');
    ylabel('Concentration');
    title('State Trajectories');
    legend('Interpreter', 'latex', 'Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);

    % Phase plane
    subplot(3, 2, 2);
    plot(stateHistory(:,1), stateHistory(:,2), 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
    hold on;
    scatter(config.xA(1), config.xA(2), 100, 'r', 'filled', 'DisplayName', '$x_A$');
    scatter(config.xB(1), config.xB(2), 100, 'g', 'filled', 'DisplayName', '$x_B$');
    scatter(config.xS(1), config.xS(2), 100, 'b', 'filled', 'DisplayName', '$x_S$');
    scatter(stateHistory(1,1), stateHistory(1,2), 80, 'k', 'o', 'LineWidth', 2, 'DisplayName', 'Start');
    scatter(stateHistory(end,1), stateHistory(end,2), 80, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'End');
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title('Phase Plane Trajectory');
    legend('Interpreter', 'latex', 'Location', 'best');
    grid on;
    axis equal;
    set(gca, 'FontSize', 12);

    % Control input
    subplot(3, 2, 3);
    plot(t, actionHistory, 'LineWidth', 2, 'Color', [0.8, 0.4, 0.2]);
    yline(0, '--k', 'LineWidth', 1);
    xlabel('Time [min]');
    ylabel('Control Input $u$', 'Interpreter', 'latex');
    title('Control Signal');
    grid on;
    set(gca, 'FontSize', 12);

    % Distance to saddle
    subplot(3, 2, 4);
    distToSaddle = vecnorm(stateHistory - config.xS', 2, 2);
    plot(t, distToSaddle, 'LineWidth', 2, 'Color', [0.2, 0.7, 0.3]);
    hold on;
    yline(config.saddleRadius, '--r', 'LineWidth', 1.5, 'DisplayName', 'Target Region');
    xlabel('Time [min]');
    ylabel('Distance');
    title('Distance to Saddle $\|x - x_S\|$', 'Interpreter', 'latex');
    legend('Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);

    % Normalized error
    subplot(3, 2, 5);
    plot(t, errorNorm, 'LineWidth', 2, 'Color', [0.6, 0.2, 0.7]);
    xlabel('Time [min]');
    ylabel('$\|\tilde{e}(t)\|$', 'Interpreter', 'latex');
    title('Normalized Error');
    grid on;
    set(gca, 'FontSize', 12);

    % Cumulative reward
    subplot(3, 2, 6);
    cumReward = cumsum(rewardHistory);
    plot(t, cumReward, 'LineWidth', 2, 'Color', [0.3, 0.5, 0.7]);
    xlabel('Time [min]');
    ylabel('Cumulative Reward');
    title('Reward Accumulation');
    grid on;
    set(gca, 'FontSize', 12);

    sgtitle('Validation Results (Greedy Policy)', 'FontSize', 16, 'FontWeight', 'bold');
end



% 
% %caso robustezza
% 
% clearvars; close all; clc;
% 
% 
% cfg = struct();
% 
% 
% cfg.dt = 0.05;
% 
% % Equilibria
% cfg.xA = [7.87298; 0.12702];
% cfg.xB = [0.12702; 7.87298];
% cfg.xS = [1.92335; 1.92335];
% 
% 
% cfg.xMax = 10.0;
% cfg.u_min = -10;
% cfg.u_max = 30; %50
% 
% 
% cfg.Nep = 3000;
% cfg.Nsteps = 1000;
% 
% % Success condition (hold near saddle)
% cfg.saddleRadius = 0.20;
% cfg.holdSteps = 40;
% cfg.successBonus = 50;
% 
% % Discretization 
% cfg.fineHalfWidth = 3.0;
% cfg.stepFine = 0.5;
% cfg.stepCoarse = 1.25;
% 
% cfg.nActions = 7;
% cfg.bin_u = linspace(cfg.u_min, cfg.u_max, cfg.nActions);
% 
% % Q-learning
% cfg.gamma = 0.90;
% cfg.alpha = 0.80;
% 
% % Exploration
% cfg.epsilon0 = 0.20;
% cfg.epsilonMin = 0.05;
% cfg.epsDecay = 0.9995;
% 
% % Curriculum
% cfg.Ncurr = 2500;
% 
% 
% cfg.uncertainty_enable = true;   % param uncertainty (per-episode) true per attiv
% cfg.dist_enable = true;          % additive state disturbance (per-step) true x attiv
% 
% % Uncertainty levels
% cfg.d_alphaX = 0.05;
% cfg.d_beta   = 0.05;
% cfg.d_alphaY = 0.05;
% 
% 
% % Disturbance levels
% cfg.dist_sigma = 0.002;
% cfg.dist_clip  = 0.01;
% 
% % Chattering reduction: penalty on delta-u
% cfg.rho_du = 2e-2;    
% cfg.du_scale = 10;
% 
% % Reward parameters
% cfg.kp = 2.0;
% cfg.rho_u = 1e-3;
% cfg.u_scale = 30;
% 
% % Build bins
% cfg.bin_x1 = nonuniformBins(0, cfg.xMax, cfg.xS(1), cfg.fineHalfWidth, cfg.stepFine, cfg.stepCoarse);
% cfg.bin_x2 = nonuniformBins(0, cfg.xMax, cfg.xS(2), cfg.fineHalfWidth, cfg.stepFine, cfg.stepCoarse);
% 
% nS1 = numel(cfg.bin_x1);
% nS2 = numel(cfg.bin_x2);
% Q = zeros(nS1, nS2, cfg.nActions);
% 
% scores = zeros(cfg.Nep,1);
% epsilon = cfg.epsilon0;
% 
% 
% for ep = 1:cfg.Nep
% 
%     lambda = min(1, ep/cfg.Ncurr);
% 
%     % reset options
%     resetOpt = struct();
%     resetOpt.sigma0 = 0.05;
%     resetOpt.xMax   = cfg.xMax;
%     resetOpt.uncertainty_enable = cfg.uncertainty_enable;
%     resetOpt.alphaX_nom = 10;
%     resetOpt.beta_nom   = 0.25;
%     resetOpt.n_nom      = 2;
%     resetOpt.alphaY_nom = 10;
%     resetOpt.d_alphaX = cfg.d_alphaX;
%     resetOpt.d_beta   = cfg.d_beta;
%     resetOpt.d_alphaY = cfg.d_alphaY;
% 
%     [~, x, epParams] = toggleReset(lambda, resetOpt);
% 
%     score = 0;
%     holdCounter = 0;
% 
%     u_prev = 0; % for delta-u penalty
% 
%     for k = 1:cfg.Nsteps
% 
%         sIdx = discretizeState(x, cfg);
% 
%         % epsilon-greedy
%         if rand() > epsilon
%             [~, aIdx] = max(Q(sIdx(1), sIdx(2), :));
%         else
%             aIdx = randi(cfg.nActions);
%         end
%         u = cfg.bin_u(aIdx);
% 
%         % step params (merge epParams + step settings)
%         stepParams = epParams;
%         stepParams.dt = cfg.dt;
%         stepParams.u_min = cfg.u_min;
%         stepParams.u_max = cfg.u_max;
% 
%         stepParams.kp = cfg.kp;
%         stepParams.rho_u = cfg.rho_u;
%         stepParams.u_scale = cfg.u_scale;
% 
%         stepParams.rho_du = cfg.rho_du;
%         stepParams.du_scale = cfg.du_scale;
%         stepParams.u_prev = u_prev;
% 
%         stepParams.dist_enable = cfg.dist_enable;
%         stepParams.dist_sigma  = cfg.dist_sigma;
%         stepParams.dist_clip   = cfg.dist_clip;
% 
%         [xNext, r, done, ~, info] = toggleStep(u, x, stepParams);
% 
%         % success hold logic
%         if norm(xNext - cfg.xS,2) <= cfg.saddleRadius
%             holdCounter = holdCounter + 1;
%             if holdCounter >= cfg.holdSteps
%                 r = r + cfg.successBonus;
%                 done = true;
%             end
%         else
%             holdCounter = 0;
%         end
% 
%         % Q update
%         sNextIdx = discretizeState(xNext, cfg);
%         q = Q(sIdx(1), sIdx(2), aIdx);
%         qNext = max(Q(sNextIdx(1), sNextIdx(2), :));
%         Q(sIdx(1), sIdx(2), aIdx) = q + cfg.alpha*(r + cfg.gamma*qNext - q);
% 
%         % iterate
%         x = xNext;
%         u_prev = info.u;   % IMPORTANT: update u_prev from saturated u
%         score = score + r;
% 
%         if done, break; end
%     end
% 
%     scores(ep) = score;
%     epsilon = max(cfg.epsilonMin, epsilon*cfg.epsDecay);
% 
%     if mod(ep,100)==0
%         fprintf("Episode %d/%d | avg score (last100)=%.2f | eps=%.3f\n", ...
%             ep, cfg.Nep, mean(scores(max(1,ep-99):ep)), epsilon);
%     end
% end
% 
% 
% resetOpt = struct();
% resetOpt.sigma0 = 0.0;
% resetOpt.xMax   = cfg.xMax;
% resetOpt.uncertainty_enable = cfg.uncertainty_enable; % test robustness if true
% resetOpt.alphaX_nom = 10; resetOpt.beta_nom=0.25; resetOpt.n_nom=2; resetOpt.alphaY_nom=10;
% resetOpt.d_alphaX = cfg.d_alphaX; resetOpt.d_beta=cfg.d_beta; resetOpt.d_alphaY=cfg.d_alphaY;
% 
% [~, x, epParams] = toggleReset(1.0, resetOpt); % start from xB (lambda=1)
% 
% Xhist = zeros(cfg.Nsteps,2);
% Uhist = zeros(cfg.Nsteps,1);
% Rhist = zeros(cfg.Nsteps,1);
% 
% Xhist(1,:) = x(:).';
% u_prev = 0;
% 
% holdCounter = 0;
% for k = 2:cfg.Nsteps
%     sIdx = discretizeState(x, cfg);
%     [~, aIdx] = max(Q(sIdx(1), sIdx(2), :));
%     u = cfg.bin_u(aIdx);
% 
%     stepParams = epParams;
%     stepParams.dt = cfg.dt;
%     stepParams.u_min = cfg.u_min;
%     stepParams.u_max = cfg.u_max;
% 
%     stepParams.kp = cfg.kp;
%     stepParams.rho_u = cfg.rho_u;
%     stepParams.u_scale = cfg.u_scale;
% 
%     stepParams.rho_du = cfg.rho_du;
%     stepParams.du_scale = cfg.du_scale;
%     stepParams.u_prev = u_prev;
% 
%     stepParams.dist_enable = cfg.dist_enable;
%     stepParams.dist_sigma  = cfg.dist_sigma;
%     stepParams.dist_clip   = cfg.dist_clip;
% 
%     [xNext, r, done, ~, info] = toggleStep(u, x, stepParams);
% 
%     if norm(xNext - cfg.xS,2) <= cfg.saddleRadius
%         holdCounter = holdCounter + 1;
%         if holdCounter >= cfg.holdSteps
%             r = r + cfg.successBonus;
%             done = true;
%         end
%     else
%         holdCounter = 0;
%     end
% 
%     Xhist(k,:) = xNext(:).';
%     Uhist(k) = info.u;
%     Rhist(k) = r;
% 
%     x = xNext;
%     u_prev = info.u;
% 
%     if done
%         Xhist = Xhist(1:k,:);
%         Uhist = Uhist(1:k);
%         Rhist = Rhist(1:k);
%         break;
%     end
% end
% 
% 
% %% --- PLOTS (time in minutes) ---
% Nsim = size(Xhist,1);
% t_min = (0:Nsim-1) * cfg.dt;  
% 
% % Normalized error calculation
% e1n = (Xhist(:,1) - cfg.xS(1)) / cfg.xS(1);
% e2n = (Xhist(:,2) - cfg.xS(2)) / cfg.xS(2);
% en  = sqrt(e1n.^2 + e2n.^2);
% 
% % ---------------------------------------------------------
% % FIGURA 1: Performance di Apprendimento
% % ---------------------------------------------------------
% figure('Color','w', 'Name', '1. Training & Rewards');
% 
% % Plot 1: Learning Curve
% subplot(2,1,1);
% plot(movmean(scores,200), "LineWidth",2);
% xlabel("Episode"); ylabel("Cumulative reward"); 
% title("Learning curve"); 
% grid on;
% 
% % Plot 2: Reward Accumulation (Validation)
% subplot(2,1,2);
% plot(t_min, cumsum(Rhist), "LineWidth",2, 'Color', '#7E2F8E'); % Viola
% xlabel("Time [min]"); ylabel("Cumulative reward"); 
% title("Reward accumulation (Validation)");
% grid on;
% 
% % ---------------------------------------------------------
% % FIGURA 2: Dinamica del Sistema
% % ---------------------------------------------------------
% figure('Color','w', 'Name', '2. System Dynamics');
% 
% % Plot 1: State Trajectories
% subplot(2,1,1);
% plot(t_min, Xhist(:,1), "LineWidth",2, 'DisplayName', 'x_1'); hold on;
% plot(t_min, Xhist(:,2), "LineWidth",2, 'DisplayName', 'x_2');
% yline(cfg.xS(1), '--k', 'DisplayName', 'Target');
% xlabel("Time [min]"); ylabel("Concentration"); 
% title("State trajectories");
% legend("Location","best"); 
% grid on;
% 
% % Plot 2: Phase Plane
% subplot(2,1,2);
% plot(Xhist(:,1), Xhist(:,2), "LineWidth",2, 'Color', 'k'); hold on;
% scatter(cfg.xA(1), cfg.xA(2), 80, 'r', 'filled', 'DisplayName', 'x_A');
% scatter(cfg.xB(1), cfg.xB(2), 80, 'b', 'filled', 'DisplayName', 'x_B');
% scatter(cfg.xS(1), cfg.xS(2), 80, 'g', 'filled', 'DisplayName', 'x_S');
% xlabel("x_1"); ylabel("x_2"); 
% title("Phase plane trajectory");
% legend("traj","x_A","x_B","x_S","Location","best"); 
% grid on; axis square;
% 
% % ---------------------------------------------------------
% % FIGURA 3: Controllo ed Errore
% % ---------------------------------------------------------
% figure('Color','w', 'Name', '3. Control & Error');
% 
% % Plot 1: Control Input
% subplot(2,1,1);
% stairs(t_min, Uhist, "LineWidth",2, 'Color', '#0072BD'); % Blu standard
% xlabel("Time [min]"); ylabel("Input u"); 
% title("Control input");
% grid on;
% 
% % Plot 2: Normalized Error
% subplot(2,1,2);
% plot(t_min, en, "LineWidth",2, 'Color', '#D95319'); % Arancione
% xlabel("Time [min]"); ylabel("||e~(t)||"); 
% title("Normalized error");
% grid on;
% %% --- helpers ---
% function idx = discretizeState(x, cfg)
% [~, i1] = min(abs(x(1) - cfg.bin_x1));
% [~, i2] = min(abs(x(2) - cfg.bin_x2));
% idx = [i1 i2];
% end
% 
% function bins = nonuniformBins(xmin, xmax, xref, halfWidth, hFine, hCoarse)
% leftEnd = max(xmin, xref - halfWidth);
% rightStart = min(xmax, xref + halfWidth);
% b1 = xmin:hCoarse:leftEnd;
% b2 = leftEnd:hFine:rightStart;
% b3 = rightStart:hCoarse:xmax;
% bins = unique([b1 b2 b3]);
% if bins(end) < xmax, bins(end+1) = xmax; end
% if bins(1) > xmin, bins = [xmin bins]; end
% end
