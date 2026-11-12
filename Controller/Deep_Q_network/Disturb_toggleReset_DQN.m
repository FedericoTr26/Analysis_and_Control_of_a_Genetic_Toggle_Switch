function [InitialObs, InitialState] = Disturb_toggleReset_DQN()

    
    % Nominal Start point (xB)
    x0_nom = [0.12702; 7.87298];
    
    % Aggiunge un rumore gaussiano 
    % per evitare che l'agente impari a memoria un solo percorso.
    x0 = x0_nom + 0.05 * randn(2,1);
    
    % Safety bounds
    xMax = 10;
    x0 = max(0, min(x0, xMax));
    
    InitialState = x0;
    InitialObs   = x0;
end