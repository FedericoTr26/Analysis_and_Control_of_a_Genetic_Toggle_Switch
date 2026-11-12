clearvars; clc; close all;


p.alpha_Y = 10;  p.alpha_Z = 10;
p.beta_Y  = 0.25; p.beta_Z = 0.25;
p.gamma_Y = 1;   p.gamma_Z = 1;
p.K_Y = 1;       p.K_Z = 1;
p.n   = 2;
p.u   = 0;       % open-loop

% Equilibria (Stable Nodes)
Nodes = [0.12702, 7.87298;   % Node A: low x1, high x2
         7.87298, 0.12702];  % Node B: high x1, low x2
         
NodeNames = {'Node A (Low-High)', 'Node B (High-Low)'};

script_path = fileparts(mfilename('fullpath')); 

% Se lo script non è salvato, usa la cartella corrente
if isempty(script_path)
    script_path = pwd; 
end

% Definisci la cartella di output relativa alla posizione dello script
out_dir = fullfile(script_path, 'fig_lyap'); 

% Crea la cartella se non esiste
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Creata cartella: %s\n', out_dir);
else
    fprintf('Cartella output trovata: %s\n', out_dir);
end

deltaJ   = 1e-6;         
Q        = eye(2);       
span     = 2.5;          
grid_res = 0.05;         
clip_to_positive = true; 
save_figs = true;        

%% 3) Loop over equilibria
for k = 1:size(Nodes,1)
    x_eq  = Nodes(k,:).';      
    name  = NodeNames{k};
    
    fprintf('\n--- %s ---\n', name);
    
    % Jacobian & Eigenvalues
    J = numerical_jacobian(@(x) sys_dynamics(x,p), x_eq, deltaJ);
    P = lyap(J', Q);
    P = 0.5*(P + P');  
    
    % Grid Generation
    x1_min = max(x_eq(1) - span, 0);  x1_max = x_eq(1) + span;
    x2_min = max(x_eq(2) - span, 0);  x2_max = x_eq(2) + span;
    
    x1_vec = x1_min : grid_res : x1_max;
    x2_vec = x2_min : grid_res : x2_max;
    [X1, X2] = meshgrid(x1_vec, x2_vec);
    
    V_vals     = zeros(size(X1));
    Vdot_vals  = zeros(size(X1));
    
    % Eval Loop
    for r = 1:size(X1,1)
        for c = 1:size(X1,2)
            x_curr = [X1(r,c); X2(r,c)];
            dx = x_curr - x_eq;
            V_vals(r,c) = dx.' * P * dx;
            
            f_curr = sys_dynamics(x_curr, p);
            gradV  = 2 * dx.' * P;          
            Vdot_vals(r,c) = gradV * f_curr; 
        end
    end
    
    %% Plotting
    fig = figure('Color','w', 'Name',['Lyapunov Analysis - ' name], ...
        'Position', [100 + 50*(k-1), 100 + 50*(k-1), 1100, 500]);
    
    t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    title(t, ['Open-loop Lyapunov analysis around ' name], 'FontSize', 14, 'Interpreter', 'none');
    
    % Plot V(x)
    nexttile;
    surf(X1, X2, V_vals, 'EdgeColor','none', 'FaceAlpha',0.9);
    hold on; grid on;
    plot3(x_eq(1), x_eq(2), 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor','k');
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    zlabel('$V(x)$','Interpreter','latex'); title('Candidate $V(x)$','Interpreter','latex');
    view(-35, 25); axis tight; colormap(gca, jet); colorbar;
    
    % Plot Vdot(x)
    nexttile;
    surf(X1, X2, Vdot_vals, 'EdgeColor','none', 'FaceAlpha',0.85);
    hold on; grid on;
    mesh(X1, X2, zeros(size(X1)), 'FaceColor','none', 'EdgeColor',[0.5 0.5 0.5], 'LineStyle',':');
    contour3(X1, X2, Vdot_vals, [0 0], 'k', 'LineWidth', 3);
    plot3(x_eq(1), x_eq(2), 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor','k');
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    zlabel('$\dot V(x)$','Interpreter','latex'); 
    title({'$\dot V(x)$', 'Black curve: $\dot V(x)=0$'}, 'Interpreter','latex');
    view(-35, 25); axis tight; colormap(gca, flipud(jet)); 
    
    %% Export
    if save_figs
        % Pulizia nome file
        safe_name = name;
        safe_name = strrep(safe_name, ' ', '_'); 
        safe_name = strrep(safe_name, '/', '-'); % Sostituisci / con - (sicuro)
        safe_name = strrep(safe_name, '\', '-'); 
        safe_name = strrep(safe_name, '(', '');  
        safe_name = strrep(safe_name, ')', '');
        
        % Costruisce percorso completo usando la cartella dello script
        fname = fullfile(out_dir, sprintf('Lyap_Analysis_%s.pdf', safe_name));
        
        try
            exportgraphics(fig, fname, 'Resolution', 300);
            fprintf('Saved successfully: %s\n', fname);
        catch ME
            fprintf('Error saving file: %s\n', ME.message);
        end
    end
end


function J = numerical_jacobian(f, x0, delta)
    n = numel(x0);
    f0 = f(x0);
    J  = zeros(n,n);
    for i = 1:n
        x1 = x0; x1(i) = x1(i) + delta;
        J(:,i) = (f(x1) - f0) / delta;
    end
end

function dx = sys_dynamics(x, p)
    x1 = x(1); x2 = x(2);
    H_Z = 1 / (1 + (x2/p.K_Z)^p.n);
    H_Y = 1 / (1 + (x1/p.K_Y)^p.n);
    D = 1 + p.beta_Y*H_Z + p.beta_Z*H_Y;
    dx1 = ((p.alpha_Y + p.u) * H_Z / D) - p.gamma_Y*x1;
    dx2 = ( p.alpha_Z        * H_Y / D) - p.gamma_Z*x2;
    dx = [dx1; dx2];
end