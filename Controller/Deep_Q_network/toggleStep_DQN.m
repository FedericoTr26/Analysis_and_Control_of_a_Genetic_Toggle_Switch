function [NextObs, Reward, IsDone, NextState] = toggleStep_DQN(ActionIndex, State)

    alphaX     = 10;
    alphaY_nom = 10;
    beta       = 0.25;
    n          = 2;
    
    dt = 0.5;   
    
  
    xS = [1.92335; 1.92335];
    
   
    % MUST match the main script definition
    actionValues = linspace(-10, 30, 7);   
    u = actionValues(ActionIndex);
    
   
    u_min = -10; u_max = 30;
    u = max(min(u, u_max), u_min);
    
    % Effective alphaY
    alphaY_act = max(alphaY_nom + u, 0);
    
  
    % Nested function for ODE
    function dx = toggle_rhs_local(x, alphaX, alphaY_act, beta, n)
        x1 = x(1); x2 = x(2);
        D  = 1 + beta/(1 + x2^n) + beta/(1 + x1^n);
        dx1 = (alphaY_act/(1 + x2^n))/D - x1;
        dx2 = (alphaX    /(1 + x1^n))/D - x2;
        dx  = [dx1; dx2];
    end

    % ---- RK4 integration ----
    x = State(:);
    k1 = toggle_rhs_local(x, alphaX, alphaY_act, beta, n);
    k2 = toggle_rhs_local(x + 0.5*dt*k1, alphaX, alphaY_act, beta, n);
    k3 = toggle_rhs_local(x + 0.5*dt*k2, alphaX, alphaY_act, beta, n);
    k4 = toggle_rhs_local(x + dt*k3, alphaX, alphaY_act, beta, n);
    
    xNext = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    
    
    xMax = 10;
    xNext = max(0, min(xNext, xMax));
    
    NextState = xNext;
    NextObs   = xNext;
    
    %Reward 
    e = xNext - xS;
    dist = norm(e,2);
    
    % Distance penalty + small effort penalty
    effort_cost = 1e-3*(u/30)^2;
    Reward = -(dist^2) - effort_cost;
    
    % Bonus for being close (Reward shaping)
    if dist < 0.20
        Reward = Reward + 1; 
    end
    
    % IsDone is false by default so agent continues to hold position
    IsDone = false; 
    
  
    if any(xNext > xMax-1e-6)
        IsDone = true;
        Reward = Reward - 50; 
    end
end