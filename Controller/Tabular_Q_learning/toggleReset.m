%la prima funzione è nel caso nominale. la seconda(commentata) ha
%incertezza parametrica per episodio

function [InitialObservation, InitialState] = toggleReset(lambda, options)

%% Input validation and default values
if nargin < 1 || isempty(lambda)
    lambda = 1.0; % Default: start from stable node xB
end

if nargin < 2
    options = struct();
end

% Validate and clamp lambda to [0,1]
if ~isscalar(lambda) || ~isnumeric(lambda)
    error('toggleReset:InvalidLambda', 'Lambda must be a numeric scalar');
end
lambda = max(0, min(1, lambda));

% Extract options with defaults
sigma0 = getOption(options, 'sigma0', 0.05);
xMax = getOption(options, 'xMax', 10.0);

% Set random seed if provided (for reproducibility)
if isfield(options, 'seed')
    rng(options.seed);
end

%% Define equilibrium points

xA = [7.87298; 0.12702];  % Stable node (undesired switching)
xB = [0.12702; 7.87298];  % Stable node (starting point)
xS = [1.92335; 1.92335];  % Saddle point (control target)

%% Compute initial state with curriculum learning
% Linear interpolation between saddle and stable node
% This enables progressive learning: balancing first, then reaching
x0 = (1 - lambda) * xS + lambda * xB;

% Add Gaussian noise for robustness
% Simulates uncertainty in biological initial conditions
if sigma0 > 0
    noise = sigma0 * randn(2, 1);
    x0 = x0 + noise;
end

% Apply physical constraints: concentrations must satisfy 0 ≤ xi ≤ xMax
x0 = max(0, min(x0, xMax));

%% Prepare outputs
InitialState = x0;
InitialObservation = InitialState; % Full observability assumption

end

%% Helper function
function value = getOption(options, fieldName, defaultValue)
% getOption - Safely extract field from options structure
    if isfield(options, fieldName)
        value = options.(fieldName);
    else
        value = defaultValue;
    end
end



% 
% %incertezza
% 
% function [InitialObservation, InitialState, epParams] = toggleReset(lambda, options)
% 
% if nargin < 1 || isempty(lambda), lambda = 1.0; end
% if nargin < 2, options = struct(); end
% 
% lambda = max(0, min(1, lambda));
% 
% % ----- default options -----
% sigma0 = getOption(options, 'sigma0', 0.05);     % init state noise
% xMax   = getOption(options, 'xMax', 10.0);
% 
% % Param uncertainty switches
% uncEnable = getOption(options, 'uncertainty_enable', false);
% 
% % Nominal parameters
% alphaX_nom   = getOption(options, 'alphaX_nom', 10);
% beta_nom     = getOption(options, 'beta_nom', 0.25);
% n_nom        = getOption(options, 'n_nom', 2);
% alphaY_nom   = getOption(options, 'alphaY_nom', 10);
% 
% % Relative uncertainty bounds (e.g., 0.1 -> +/-10%)
% d_alphaX = getOption(options, 'd_alphaX', 0.10);
% d_beta   = getOption(options, 'd_beta', 0.10);
% d_alphaY = getOption(options, 'd_alphaY', 0.10);
% 
% 
% if isfield(options, 'seed')
%     rng(options.seed);
% end
% 
% % Equilibria (fixed reference)
% xA = [7.87298; 0.12702];
% xB = [0.12702; 7.87298];
% xS = [1.92335; 1.92335];
% 
% % Curriculum start state
% x0 = (1 - lambda) * xS + lambda * xB;
% 
% % Add initial condition noise
% if sigma0 > 0
%     x0 = x0 + sigma0 * randn(2,1);
% end
% x0 = max(0, min(x0, xMax));
% 
% InitialState = x0;
% InitialObservation = InitialState;
% 
% 
% epParams = struct();
% epParams.xA = xA; epParams.xB = xB; epParams.xS = xS;
% epParams.xMax = xMax;
% 
% epParams.alphaX = alphaX_nom;
% epParams.beta   = beta_nom;
% epParams.n      = n_nom;
% epParams.alphaY_nom = alphaY_nom;
% 
% if uncEnable
%     epParams.alphaX    = alphaX_nom  * (1 + d_alphaX*(2*rand-1));
%     epParams.beta      = beta_nom    * (1 + d_beta  *(2*rand-1));
%     epParams.alphaY_nom= alphaY_nom  * (1 + d_alphaY*(2*rand-1));
% end
% end
% 
% function value = getOption(options, fieldName, defaultValue)
% if isfield(options, fieldName), value = options.(fieldName);
% else, value = defaultValue;
% end
% end
