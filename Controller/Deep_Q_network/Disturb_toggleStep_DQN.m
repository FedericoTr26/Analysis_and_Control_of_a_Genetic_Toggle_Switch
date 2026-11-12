function [NextObs, Reward, IsDone, NextState] = Disturb_toggleStep_DQN(ActionIndex, State)

    % Nominal parameters
    alphaX_nom = 10;
    beta_nom   = 0.25;
    n          = 2;
    
    % Introduciamo variabilità (+/- 10% variazione stocastica ad ogni step)
    p_var = 0.10; 
    alphaX = alphaX_nom * (1 + p_var * randn());
    beta   = beta_nom   * (1 + p_var * randn());
    
    % Assicuriamo che i parametri restino positivi
    alphaX = max(0, alphaX);
    beta   = max(0, beta);

   
    dt = 0.5;   % [s]
    
    
    xS = [1.92335; 1.92335];
    
  
    % Discrete action set
    actionValues = linspace(-10, 30, 7);   
    u_nominal = actionValues(ActionIndex);
    
    % Aggiungiamo rumore all'attuatore (Input Disturbance)
    %  deviazione standard di 1.0 unità
    u_noise_std = 1.0; 
    u_disturbed = u_nominal + u_noise_std * randn();
    
  
    % La saturazione avviene DOPO il disturbo
    u_min = -10; u_max = 30;
    u = max(min(u_disturbed, u_max), u_min);
    
    % Effective alphaY (nominal is 10)
    alphaY_nom = 10;
    alphaY_act = max(alphaY_nom + u, 0);
    
    
    function dx = local_rhs(x, aX, aY, b, n_exp)
        x1 = x(1); x2 = x(2);
        D  = 1 + b/(1 + x2^n_exp) + b/(1 + x1^n_exp);
        dx1 = (aY/(1 + x2^n_exp))/D - x1;
        dx2 = (aX/(1 + x1^n_exp))/D - x2;
        dx  = [dx1; dx2];
    end

 
    x = State(:);
    k1 = local_rhs(x, alphaX, alphaY_act, beta, n);
    k2 = local_rhs(x + 0.5*dt*k1, alphaX, alphaY_act, beta, n);
    k3 = local_rhs(x + 0.5*dt*k2, alphaX, alphaY_act, beta, n);
    k4 = local_rhs(x + dt*k3, alphaX, alphaY_act, beta, n);
    
    xNext = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    

    xMax = 10;
    xNext = max(0, min(xNext, xMax));
    
    NextState = xNext;
    NextObs   = xNext;
    
  
    e = xNext - xS;
    dist = norm(e,2);
    
    % Penalize effort (using the actual disturbed u felt by the plant)
    effort_cost = 1e-3*(u/30)^2;
    Reward = -(dist^2) - effort_cost;
    
    if dist < 0.20
        Reward = Reward + 1; 
    end
    
   
    IsDone = false; 
    
    if any(xNext > xMax-1e-6)
        IsDone = true;
        Reward = Reward - 50; 
    end
end