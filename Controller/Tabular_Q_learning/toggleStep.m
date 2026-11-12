%la prima funzione è nominale, la seconda(commentata) è riscritta per provare a farla funzionare con un
%disturbo additivo

function [NextObs, Reward, IsDone, NextState, Info] = toggleStep(Action, State, params)

%% Input validation
if nargin < 2
    error('toggleStep:NotEnoughInputs', 'Action and State are required');
end

if nargin < 3
    params = struct();
end

% Ensure State is a column vector
State = State(:);
if length(State) ~= 2
    error('toggleStep:InvalidState', 'State must be a 2-element vector');
end

%% Extract parameters with defaults
dt = getParam(params, 'dt', 0.05);
alphaY_nom = getParam(params, 'alphaY_nom', 10);
u_min = getParam(params, 'u_min', -10);
u_max = getParam(params, 'u_max', 30);
xMax = getParam(params, 'xMax', 10.0);

% Equilibria
xA = [7.87298; 0.12702];
xB = [0.12702; 7.87298];
xS = [1.92335; 1.92335];

% Reward parameters
deltaS = getParam(params, 'deltaS', 0.20);
deltaB = getParam(params, 'deltaB', 0.30);
deltaA = getParam(params, 'deltaA', 0.30);
kp = getParam(params, 'kp', 2.0);
rho = getParam(params, 'rho', 1e-3);
reward_scale = getParam(params, 'reward_scale', 1.0);

%% Saturate control action
u = max(min(Action, u_max), u_min);

% Compute effective alphaY
alphaY_act = alphaY_nom + u;
alphaY_act = max(alphaY_act, 0); % Physical constraint: must be non-negative

%% Integrate dynamics using Runge-Kutta 4th order
% This provides better accuracy than forward Euler
f = @(x) toggle_rhs(x, alphaY_act);

k1 = f(State);
k2 = f(State + 0.5 * dt * k1);
k3 = f(State + 0.5 * dt * k2);
k4 = f(State + dt * k3);

NextState = State + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);

% Apply physical constraints
NextState = max(0, min(NextState, xMax));

%% Compute reward components
% 1. Base reward, negative normalized squared error from saddle
e_abs = NextState - xS;
e_rel = e_abs ./ xS; % Normalization
baseReward = -sum(e_rel.^2) * reward_scale;

% 2. Progress reward: encourages approaching the saddle
d_prev = norm(State - xS, 2);
d_new = norm(NextState - xS, 2);
progReward = kp * (d_prev - d_new);

% 3. Control penalty: discourages large control actions
u_scale = 30; % Normalization factor
ctrlPenalty = -rho * (u / u_scale)^2;

% Initialize total reward
Reward = baseReward + progReward + ctrlPenalty;

%% Termination conditions and bonuses/penalties
IsDone = false;

% Check if in saddle region (bonus)
in_saddle = (d_new <= deltaS);
if in_saddle
    Reward = Reward + 1.0;
end

% Check if stuck in xB region (penalty)
d_xB = norm(NextState - xB, 2);
in_xB = (d_xB <= deltaB);
if in_xB
    Reward = Reward - 1.0;
end

% Check if switched to undesired equilibrium xA (large penalty + termination)
d_xA = norm(NextState - xA, 2);
switched_to_xA = (d_xA <= deltaA);
if switched_to_xA
    Reward = Reward - 20.0;
    IsDone = true;
end

% Check if exceeding maximum bounds (penalty + termination)
exceeded_bounds = any(NextState >= xMax - 1e-6);
if exceeded_bounds
    Reward = Reward - 10.0;
    IsDone = true;
end

%% Prepare outputs
NextObs = NextState;

% Additional information structure
if nargout >= 5
    Info = struct();
    Info.distance_to_saddle = d_new;
    Info.distance_to_xB = d_xB;
    Info.distance_to_xA = d_xA;
    Info.in_saddle_region = in_saddle;
    Info.in_xB_region = in_xB;
    Info.switched_to_xA = switched_to_xA;
    Info.exceeded_bounds = exceeded_bounds;
    Info.effective_alphaY = alphaY_act;
    Info.saturated_control = u;

    % Reward breakdown
    Info.reward_components.base = baseReward;
    Info.reward_components.progress = progReward;
    Info.reward_components.control_penalty = ctrlPenalty;
    Info.reward_components.saddle_bonus = in_saddle * 1.0;
    Info.reward_components.xB_penalty = in_xB * (-1.0);
    Info.reward_components.xA_penalty = switched_to_xA * (-20.0);
    Info.reward_components.bounds_penalty = exceeded_bounds * (-10.0);
end

end

%% Helper functions
function dx = toggle_rhs(x, alphaY_act)

    % Fixed parameters
    alphaX = 10;  % Production rate for x2
    beta = 0.25;  % Mutual repression strength
    n = 2;        % Hill coefficient

    x1 = x(1);
    x2 = x(2);

    % Common denominator (normalization factor)
    D = 1 + beta/(1 + x2^n) + beta/(1 + x1^n);

    % State derivatives
    dx1 = (alphaY_act / (1 + x2^n)) / D - x1;
    dx2 = (alphaX / (1 + x1^n)) / D - x2;

    dx = [dx1; dx2];
end

function value = getParam(params, fieldName, defaultValue)
% getParam - Safely extract parameter from structure
    if isfield(params, fieldName)
        value = params.(fieldName);
    else
        value = defaultValue;
    end
end






% 
% 
% 
% %%disturbo
% 
% function [NextObs, Reward, IsDone, NextState, Info] = toggleStep(Action, State, params)
% % toggleStep - One RK4 step + reward + (optional) state disturbance.
% 
% if nargin < 3, params = struct(); end
% State = State(:);
% 
% % --- defaults ---
% dt = getParam(params, 'dt', 0.05);
% 
% u_min = getParam(params, 'u_min', -10);
% u_max = getParam(params, 'u_max',  30);
% 
% % Equilibria
% xA = getParam(params, 'xA', [7.87298; 0.12702]);
% xB = getParam(params, 'xB', [0.12702; 7.87298]);
% xS = getParam(params, 'xS', [1.92335; 1.92335]);
% 
% xMax = getParam(params, 'xMax', 10.0);
% 
% % Model params (can be uncertain per-episode)
% alphaX     = getParam(params, 'alphaX', 10);
% beta       = getParam(params, 'beta', 0.25);
% n          = getParam(params, 'n', 2);
% alphaY_nom = getParam(params, 'alphaY_nom', 10);
% 
% % Reward parameters
% deltaS = getParam(params, 'deltaS', 0.20);
% deltaB = getParam(params, 'deltaB', 0.30);
% deltaA = getParam(params, 'deltaA', 0.30);
% kp     = getParam(params, 'kp', 2.0);
% 
% rho_u  = getParam(params, 'rho_u', 1e-3);     % magnitude penalty on u
% u_scale= getParam(params, 'u_scale', 30);
% 
% % OPTIONAL: penalty on action variation to reduce chattering
% rho_du   = getParam(params, 'rho_du', 0.0);   % set >0 to penalize switches
% du_scale = getParam(params, 'du_scale', 10);
% u_prev   = getParam(params, 'u_prev', 0);     % passed in by training
% 
% % Disturbance on state
% dist_enable = getParam(params, 'dist_enable', false);
% dist_sigma  = getParam(params, 'dist_sigma', 0.0); % e.g. 0.01
% dist_clip   = getParam(params, 'dist_clip', 0.05); % bound the noise
% 
% % --- Saturate action and compute alphaY_act ---
% u = max(min(Action, u_max), u_min);
% alphaY_act = max(alphaY_nom + u, 0);
% 
% % --- RK4 integration ---
% f = @(x) toggle_rhs(x, alphaX, alphaY_act, beta, n);
% 
% k1 = f(State);
% k2 = f(State + 0.5*dt*k1);
% k3 = f(State + 0.5*dt*k2);
% k4 = f(State + dt*k3);
% 
% NextState = State + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
% 
% % --- Add disturbance on state (after integration) ---
% w = [0;0];
% if dist_enable && dist_sigma > 0
%     w = dist_sigma * randn(2,1);
%     % clip to avoid unrealistic jumps
%     w = max(min(w, dist_clip), -dist_clip);
%     NextState = NextState + w;
% end
% 
% % Physical bounds
% NextState = max(0, min(NextState, xMax));
% NextObs = NextState;
% 
% % --- Reward ---
% e_rel = (NextState - xS) ./ xS;
% baseReward = -sum(e_rel.^2);
% 
% d_prev = norm(State - xS, 2);
% d_new  = norm(NextState - xS, 2);
% progReward = kp * (d_prev - d_new);
% 
% ctrlPenalty = -rho_u * (u/u_scale)^2;
% 
% % penalize action changes (chattering)
% du = u - u_prev;
% switchPenalty = -rho_du * (du/du_scale)^2;
% 
% Reward = baseReward + progReward + ctrlPenalty + switchPenalty;
% 
% % Bonus/penalties + termination
% IsDone = false;
% 
% in_saddle = (d_new <= deltaS);
% if in_saddle
%     Reward = Reward + 1.0;
% end
% 
% if norm(NextState - xB,2) <= deltaB
%     Reward = Reward - 1.0;
% end
% 
% if norm(NextState - xA,2) <= deltaA
%     Reward = Reward - 20.0;
%     IsDone = true;
% end
% 
% if any(NextState >= xMax - 1e-6)
%     Reward = Reward - 10.0;
%     IsDone = true;
% end
% 
% % Info
% if nargout >= 5
%     Info = struct();
%     Info.u = u;
%     Info.u_prev = u_prev;
%     Info.du = du;
%     Info.w = w;
%     Info.dS = d_new;
%     Info.baseReward = baseReward;
%     Info.progReward = progReward;
%     Info.ctrlPenalty = ctrlPenalty;
%     Info.switchPenalty = switchPenalty;
%     Info.in_saddle = in_saddle;
% end
% end
% 
% function dx = toggle_rhs(x, alphaX, alphaY_act, beta, n)
% x1 = x(1); x2 = x(2);
% D  = 1 + beta/(1 + x2^n) + beta/(1 + x1^n);
% 
% dx1 = (alphaY_act/(1 + x2^n))/D - x1;
% dx2 = (alphaX    /(1 + x1^n))/D - x2;
% 
% dx = [dx1; dx2];
% end
% 
% function value = getParam(params, fieldName, defaultValue)
% if isfield(params, fieldName), value = params.(fieldName);
% else, value = defaultValue;
% end
% end
