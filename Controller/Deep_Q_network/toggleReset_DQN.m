function [InitialObs, InitialState] = toggleReset_DQN()


    % Start point (xB)
    x0 = [0.12702; 7.87298];

    % Safety bounds (coherent with step)
    xMax = 10;
    x0 = max(0, min(x0, xMax));

    InitialState = x0;
    InitialObs   = x0;
end


