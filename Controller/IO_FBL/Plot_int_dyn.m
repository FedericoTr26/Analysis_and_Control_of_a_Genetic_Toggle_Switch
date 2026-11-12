clearvars; clc; close all;


alphaZ = 10;
betaY  = 0.25;
betaZ  = 0.25;
gammaZ = 1;


x1d = 1.92335;                 
HYd = 1/(1 + x1d^2);           

% Initial condition (from xB)
x2_0 = 7.87298;

% Zero dynamics function: x2dot = f(x2)
f = @(x2) (alphaZ * HYd ./ (1 + betaY./(1+x2.^2) + betaZ*HYd)) - gammaZ*x2;


x2s = fzero(f, 2); 

x2_min = 0;
x2_max = 10;
x2_grid = linspace(x2_min, x2_max, 2000);
x2dot_grid = f(x2_grid);

%% ========================================================================
%  2. PLOT (Dimensioni A4, Labels Originali)
%  ========================================================================
% Imposto la figura larga 16cm e alta 10cm 
fig = figure('Color','w');
set(fig, 'Units', 'centimeters');
set(fig, 'Position', [10 10 16 10]); 

hold on; grid on;

% 1. Curva Principale
hcurve = plot(x2_grid, x2dot_grid, 'LineWidth', 1.5);

% 2. Asse orizzontale (x2dot = 0) - Escluso dalla legenda
yline(0, '-', 'LineWidth', 1.0, 'HandleVisibility','off');

% 3. Punto di Equilibrio (Target) - Pallino Pieno
heq = plot(x2s, 0, 'ko', 'MarkerFaceColor','k', 'MarkerSize', 7, 'LineWidth', 1.5);

% 4. Condizione Iniziale - Pallino Vuoto
hinit = plot(x2_0, 0, 'ko', 'MarkerFaceColor','w', 'MarkerSize', 7, 'LineWidth', 1.5);

% 5. Linea tratteggiata verticale - Esclusa dalla legenda
x2dot_0 = f(x2_0);
plot([x2_0 x2_0], [0 x2dot_0], 'k--', 'LineWidth', 1.0, 'HandleVisibility','off');

% 6. Frecce di direzione (Vector Field) - Escluse dalla legenda
knots = sort([x2_min, x2s, x2_max]); 
for i = 1:(numel(knots)-1)
    xm = 0.5*(knots(i)+knots(i+1));
    s  = sign(f(xm));
    if s == 0, continue; end
    
    dx = 0.35*(knots(i+1)-knots(i));
    dx = max(min(dx, 0.5), 0.15);
    
    if s > 0
        quiver(xm-dx/2, 0, dx, 0, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2, ...
               'HandleVisibility','off');
    else
        quiver(xm+dx/2, 0, -dx, 0, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2, ...
               'HandleVisibility','off');
    end
end


xlabel('$x_2$', 'Interpreter','latex', 'FontSize', 14);
ylabel('$\dot{x}_2$', 'Interpreter','latex', 'FontSize', 14);
title('Internal Dynamics', 'Interpreter','latex', 'FontSize', 16);


legend([hcurve, heq, hinit], ...
       {'$\dot{x}_2$', '$x_{2,zd}^\ast$', '$x_2(0)$'}, ...
       'Interpreter','latex', 'Location','northeast', 'FontSize', 12);

% Limiti assi
yl = ylim;
ylim([min(yl(1), min(x2dot_grid)*1.1) max(yl(2), max(x2dot_grid)*1.1)]);
xlim([x2_min, x2_max]);


fileName = 'Internal_Dynamics.pdf';
fullPath = fullfile(pwd, fileName);

try
    exportgraphics(fig, fullPath, 'ContentType', 'vector');
    fprintf('Plot salvato in: %s\n', fullPath);
catch
    print(fig, fullPath, '-dpdf', '-r300', '-bestfit');
end