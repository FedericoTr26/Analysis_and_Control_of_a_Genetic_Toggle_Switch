clearvars; close; clc;


x1_range = [0 15];
x2_range = [0 15];
gridN    = 9;          % number of grid points per axis (increase if needed)
tolUniq  = 1e-5;       % uniqueness tolerance for equilibria merging

% fsolve options
opts = optimoptions('fsolve', ...
    'Display','off', ...
    'FunctionTolerance',1e-12, ...
    'StepTolerance',1e-12, ...
    'OptimalityTolerance',1e-12, ...
    'MaxIterations',500, ...
    'MaxFunctionEvaluations',5000);

cases = struct([]);

% Case I (bistable) - the one that gives 3 equilibria in your pplane8 test
cases(1).name   = 'Case I: balanced (expected 3 equilibria)';
cases(1).p      = makeParams( ...
    10, 10, ...        % alphaY, alphaZ
    0.25, 0.25, ...    % betaY, betaZ
    1, 1, ...          % gammaY, gammaZ
    1, 1);             % KY, KZ

% Case II (monostable) - the one you used to get a single equilibrium
cases(2).name   = 'Case II: unbalanced decay (expected 1 equilibrium)';
cases(2).p      = makeParams( ...
    13, 3, ...
    0.3, 0.3, ...
    3, 1, ...
    1, 1);

% 
% xi1'=((alphaY/(1+(x2/KZ)^2))/1+ betaY/(1+(x2/KZ)^2)+betaZ/(1+(x1/KY)^2)))-gammaY*x1;
% xi2'=((alphaZ/(1+(x1/KY)^2))/(1+betaY/(1+(x2/KZ)^2)+betaZ/(1+(x1/KY)^2)))-gammaZ*x2;
% 
% ((alphaY/(1+(x2)^2))/(1+ betaY/(1+(x2)^2)+betaZ/(1+(x1)^2)))-gammaY*x1;
% 
% ((alphaZ/(1+(x1)^2))/(1+betaY/(1+(x2)^2)+betaZ/(1+(x1)^2)))-gammaZ*x2;
% alphaY,alphaZ,betaY,betaZ,gammaY,gammaZ,KY,KZ

for k = 1:numel(cases)
    fprintf('\n============================================================\n');
    fprintf('%s\n', cases(k).name);
    fprintf('Parameters:\n');
    disp(cases(k).p);

    % 1) Find equilibria numerically: solve f(x)=0 from many initial guesses
    eqs = findEquilibria(@(x) f_model(x, cases(k).p), ...
                         x1_range, x2_range, gridN, opts, tolUniq);

    if isempty(eqs)
        fprintf('No equilibria found in the search box.\n');
        continue;
    end

    
    eqs = sortrows(eqs, [1 2]);

    fprintf('Found %d equilibrium point(s) in [%g,%g]x[%g,%g]:\n', ...
        size(eqs,1), x1_range(1), x1_range(2), x2_range(1), x2_range(2));

    
    for i = 1:size(eqs,1)
        xeq = eqs(i,:).';
        J   = jacobian_analytic(xeq, cases(k).p);
        trJ = trace(J);
        detJ= det(J);
        ev  = eig(J);

        typeStr = classifyEquilibrium(trJ, detJ, ev);

        fprintf('\n-- Equilibrium #%d\n', i);
        fprintf('x* = [%.8f; %.8f]\n', xeq(1), xeq(2));
        fprintf('J(x*) = \n');
        disp(J);
        fprintf('trace(J) = %.8f\n', trJ);
        fprintf('det(J)   = %.8f\n', detJ);
        fprintf('eigs(J)  = [%.8f%+.8fi, %.8f%+.8fi]\n', ...
            real(ev(1)), imag(ev(1)), real(ev(2)), imag(ev(2)));
        fprintf('Type: %s\n', typeStr);
    end
end

%% ================= LOCAL FUNCTIONS =================

function p = makeParams(alphaY, alphaZ, betaY, betaZ, gammaY, gammaZ, KY, KZ)
    p = struct('alphaY',alphaY,'alphaZ',alphaZ, ...
               'betaY',betaY,'betaZ',betaZ, ...
               'gammaY',gammaY,'gammaZ',gammaZ, ...
               'KY',KY,'KZ',KZ);
end

function F = f_model(x, p)
    % x = [x1; x2]
    x1 = x(1); x2 = x(2);

    % Helpful terms (keep exactly the structure of the original model)
    s = 1 + (x2/p.KZ)^2;           % 1 + (x2/KZ)^2
    t = 1 + (x1/p.KY)^2;           % 1 + (x1/KY)^2

    Phi = 1 + p.betaY/s + p.betaZ/t;

    f1 = (p.alphaY/s) / Phi - p.gammaY*x1;
    f2 = (p.alphaZ/t) / Phi - p.gammaZ*x2;

    F = [f1; f2];
end

function J = jacobian_analytic(x, p)
    % Analytic Jacobian of the model (consistent with original form).
    x1 = x(1); x2 = x(2);

    s = 1 + (x2/p.KZ)^2;
    t = 1 + (x1/p.KY)^2;
    Phi = 1 + p.betaY/s + p.betaZ/t;

    % Useful derivatives:
    % d/dx1 (1/t) = -(2 x1 / KY^2) / t^2
    d_inv_t__dx1 = -(2*x1/(p.KY^2)) / (t^2);

    % d/dx2 (1/s) = -(2 x2 / KZ^2) / s^2
    d_inv_s__dx2 = -(2*x2/(p.KZ^2)) / (s^2);

    % dPhi/dx1 = betaZ * d(1/t)/dx1
    dPhi__dx1 = p.betaZ * d_inv_t__dx1;

    % dPhi/dx2 = betaY * d(1/s)/dx2
    dPhi__dx2 = p.betaY * d_inv_s__dx2;

    % N1(x2)=alphaY/s,  N2(x1)=alphaZ/t
    N1 = p.alphaY/s;
    N2 = p.alphaZ/t;

    % dN1/dx2 = alphaY * d(1/s)/dx2
    dN1__dx2 = p.alphaY * d_inv_s__dx2;

    % dN2/dx1 = alphaZ * d(1/t)/dx1
    dN2__dx1 = p.alphaZ * d_inv_t__dx1;

    % f1 = N1/Phi - gammaY*x1
    % f2 = N2/Phi - gammaZ*x2
    %
    % Quotient rule:
    df1dx1 = -(N1/(Phi^2))*dPhi__dx1 - p.gammaY;
    df1dx2 = (dN1__dx2*Phi - N1*dPhi__dx2)/(Phi^2);

    df2dx1 = (dN2__dx1*Phi - N2*dPhi__dx1)/(Phi^2);
    df2dx2 = -(N2/(Phi^2))*dPhi__dx2 - p.gammaZ;

    J = [df1dx1, df1dx2;
         df2dx1, df2dx2];
end

function eqs = findEquilibria(fun, x1_range, x2_range, gridN, opts, tolUniq)
    % Multi-start fsolve on a grid of initial guesses, collect unique roots.
    x1g = linspace(x1_range(1), x1_range(2), gridN);
    x2g = linspace(x2_range(1), x2_range(2), gridN);

    eqs = [];

    for a = 1:numel(x1g)
        for b = 1:numel(x2g)
            x0 = [x1g(a); x2g(b)];

            try
                [xsol, fval, exitflag] = fsolve(fun, x0, opts);
            catch
                continue;
            end

            if exitflag <= 0
                continue;
            end

            if any(~isfinite(xsol)) || any(~isreal(xsol))
                continue;
            end

            % Check residual
            if norm(fval, 2) > 1e-8
                continue;
            end

            % Enforce (optional) small negative clean-up
            xsol(abs(xsol) < 1e-10) = 0;

            % Uniqueness check
            if isempty(eqs)
                eqs = xsol(:).';
            else
                d = vecnorm(eqs - xsol(:).', 2, 2);
                if all(d > tolUniq)
                    eqs = [eqs; xsol(:).']; %#ok<AGROW>
                end
            end
        end
    end
end

function typeStr = classifyEquilibrium(trJ, detJ, ev)
    % Basic 2D classification from trace/det/discriminant.
    disc = trJ^2 - 4*detJ;

    % Hyperbolicity check (no eigenvalue close to zero)
    hyper = all(abs(real(ev)) > 1e-8 | abs(imag(ev)) > 1e-8);

    if detJ < 0
        typeStr = 'Saddle (unstable)';
        return;
    end

    if detJ > 0
        if trJ < 0
            if disc > 0
                typeStr = 'Stable node';
            elseif disc < 0
                typeStr = 'Stable focus (spiral)';
            else
                typeStr = 'Stable degenerate (repeated eigenvalue)';
            end
        elseif trJ > 0
            if disc > 0
                typeStr = 'Unstable node';
            elseif disc < 0
                typeStr = 'Unstable focus (spiral)';
            else
                typeStr = 'Unstable degenerate (repeated eigenvalue)';
            end
        else
            typeStr = 'Center/critical (trace ~ 0) -> need higher-order analysis';
        end
    else
        typeStr = 'Degenerate (det ~ 0) -> non-hyperbolic, need higher-order analysis';
    end

    if ~hyper
        typeStr = [typeStr, ' [non-hyperbolic warning]']; %#ok<AGROW>
    end
end
