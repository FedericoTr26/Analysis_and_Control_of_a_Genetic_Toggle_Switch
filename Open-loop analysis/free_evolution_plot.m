function free_evolution_plot()

    clearvars; clc; close all;

    % Cartella Output
    outFolder = fullfile(pwd,'figures_free');
    if ~exist(outFolder,'dir'), mkdir(outFolder); end
    fprintf('Saving PDFs to:\n  %s\n\n', outFolder);

    %% =========================
    %  0. Impostazioni Grafiche Globali (LaTeX + Font Grandi)
    %% =========================
    set(groot, ...
        'defaultFigureColor', 'w', ...
        'defaultAxesFontSize', 16, ... 
        'defaultTextFontSize', 16, ...
        'defaultAxesLineWidth', 1.2, ...
        'defaultLineLineWidth', 2.0, ...
        'defaultTextInterpreter', 'latex', ...
        'defaultAxesTickLabelInterpreter', 'latex', ...
        'defaultLegendInterpreter', 'latex');

  
    p.alpha_Y = 10;   p.alpha_Z = 10;
    p.beta_Y  = 0.25; p.beta_Z  = 0.25;
    p.gamma_Y = 1;    p.gamma_Z = 1;
    p.K_Y     = 1;    p.K_Z     = 1;
    p.n       = 2;
    p.u       = 0;    % open-loop

  
    eqA = [0.12702; 7.87298];   % Node A (Low x1, High x2)
    eqB = [7.87298; 0.12702];   % Node B (High x1, Low x2)
    eqS = [1.92335; 1.92335];   % Saddle

   
    %  3. Condizioni Iniziali
    ICs = [ ...
        0.20  6.00;
        6.00  0.20;
        1.50  5.50;
        5.50  1.50;
        3.00  3.00;
        1.90  1.95;
        2.20  1.70;
    ];
    nIC = size(ICs,1);

  
    tspan = [0 50];
    t_eval = linspace(tspan(1), tspan(2), 2501);
    opts  = odeset('RelTol',1e-7,'AbsTol',1e-9,'MaxStep',0.05);

    T_cell = cell(nIC,1);
    X_cell = cell(nIC,1);
    legTxt = cell(nIC,1);

    fprintf('Open-loop simulations: %d initial conditions...\n', nIC);
    for k = 1:nIC
        x0 = ICs(k,:)';
        [t,x] = ode45(@(t,x) toggle_dyn(t,x,p), t_eval, x0, opts);
        T_cell{k} = t;
        X_cell{k} = x;
        legTxt{k} = sprintf('IC%d: [%.2f, %.2f]', k, ICs(k,1), ICs(k,2));
    end

    %% =========================
    %  5. PLOT x1(t)
    %% =========================
    % Per x1: eqA è basso, eqB è alto
    refVals_x1   = [eqA(1), eqB(1), eqS(1)];
    refLabels_x1 = {'Node A', 'Node B', 'Saddle'};
    
    localPlotSavePDF(T_cell, X_cell, 1, ...
        'Evolution of state $x_1(t)$', ...
        '$x_1(t)$', ...
        legTxt, ...
        refVals_x1, refLabels_x1, ...
        fullfile(outFolder, 'x1_evolution_free.pdf'));

    %% =========================
    %  6. PLOT x2(t)
    %% =========================
    % Per x2: eqA è alto, eqB è basso
    refVals_x2   = [eqA(2), eqB(2), eqS(2)];
    refLabels_x2 = {'Node A', 'Node B', 'Saddle'};

    localPlotSavePDF(T_cell, X_cell, 2, ...
        'Evolution of state $x_2(t)$', ...
        '$x_2(t)$', ...
        legTxt, ...
        refVals_x2, refLabels_x2, ...
        fullfile(outFolder, 'x2_evolution_free.pdf'));

    fprintf('Done. PDFs saved.\n');
end

%% ===== Helper Functions =====

function localPlotSavePDF(T_cell, X_cell, stateIdx, ttl, ylab, legStr, refVals, refLabs, pdfPath)
    % Crea la figura con le dimensioni specificate (Stile "Simulink visible")
    fig = figure('Visible','on', 'Units','pixels', 'Position',[120 120 980 560]);
    set(fig, 'Color','w', 'Renderer','painters');
    
    textColor = [0.12 0.18 0.26];
    ax = axes(fig, 'Color', 'w', 'XColor', textColor, 'YColor', textColor, ...
        'GridColor', textColor, 'GridAlpha', 0.14); 
    hold(ax, 'on'); grid(ax, 'on');

    colors = [0.16 0.36 0.63; 0.75 0.30 0.10; 0.04 0.52 0.48;
        0.42 0.30 0.67; 0.45 0.43 0.14; 0.30 0.49 0.59; 0.65 0.21 0.43];
    plotHandles = [];
    allData = [];

    % Plot delle curve
    for k = 1:length(T_cell)
        data = X_cell{k}(:, stateIdx);
        allData = [allData; data]; %#ok<AGROW>
        h = plot(ax, T_cell{k}, data, 'LineWidth', 2.0, 'Color', colors(k,:));
        plotHandles = [plotHandles; h]; %#ok<AGROW>
    end

    for i = 1:3
        val = refVals(i);
        lbl = refLabs{i};
        
        
        if val < 4
            vAlign = 'top';    % SOPRA la linea
        else
            vAlign = 'bottom'; % SOTTO la linea
        end
        
        % Stile: Tratteggiata per Nodi, Punteggiata per Saddle
        if strcmp(lbl, 'Saddle')
            lStyle = ':'; lColor = 'k'; lWidth = 1.5;
        else
            lStyle = '--'; lColor = [0.4 0.4 0.4]; lWidth = 1.2;
        end

        yline(val, lStyle, 'Color', lColor, 'LineWidth', lWidth, ...
            'Label', lbl, ...
            'LabelHorizontalAlignment', 'right', ... % Sempre a destra
            'LabelVerticalAlignment', vAlign, ...    % Variabile (Sopra o Sotto)
            'FontSize', 12, 'Interpreter', 'latex', ...
            'HandleVisibility', 'off'); 
    end

    % Styling Assi e Titoli
    ax.FontSize = 18;              
    ax.LineWidth = 1.2;
    
    xlabel(ax, 'Time [min]', 'Interpreter','latex', 'FontSize', 20, 'FontWeight','normal');
    ylabel(ax, ylab,         'Interpreter','latex', 'FontSize', 20, 'FontWeight','normal');
    title(ax, ttl,           'Interpreter','latex', 'FontSize', 22, 'FontWeight', 'normal'); 

    % Legenda a destra
    lgd = legend(plotHandles, legStr, 'Location', 'East'); 
    lgd.Box = 'on';
    lgd.Interpreter = 'latex';
    lgd.FontSize = 14;
    lgd.TextColor = textColor;
    lgd.Color = 'white';
    ax.Title.Color = textColor;
    ax.XLabel.Color = textColor;
    ax.YLabel.Color = textColor; 

    % Margini sicuri sull'asse Y per non tagliare le scritte
    yMin = min([allData; refVals(:)]) - 0.5;
    yMax = max([allData; refVals(:)]) + 0.5;
    ylim([max(0, yMin), yMax]);

    set(ax, 'LooseInset', max(get(ax,'TightInset'), 0.02));

    % Export
    exportgraphics(fig, pdfPath, 'ContentType','vector', 'BackgroundColor','white');
end

function dx = toggle_dyn(~, x, p)
    x1 = max(0, x(1)); 
    x2 = max(0, x(2));
    H_Z = 1 / (1 + (x2/p.K_Z)^p.n);
    H_Y = 1 / (1 + (x1/p.K_Y)^p.n);
    D = 1 + p.beta_Y*H_Z + p.beta_Z*H_Y;
    alphaY = p.alpha_Y + p.u;
    dx1 = (alphaY     * H_Z / D) - p.gamma_Y*x1;
    dx2 = (p.alpha_Z  * H_Y / D) - p.gamma_Z*x2;
    dx = [dx1; dx2];
end