clearvars; close all; clc;

modelName = 'PI_control';     
outFolder = '.';     
if ~exist(outFolder,'dir'); mkdir(outFolder); end


load_system(modelName);
out = sim(modelName);


x1_ts = out.x1;
x2_ts = out.x2;
e_ts  = out.error;
u_ts  = out.u;

% Extract time and data from timeseries
t1 = x1_ts.Time;  x1 = x1_ts.Data;
t2 = x2_ts.Time;  x2 = x2_ts.Data;
te = e_ts.Time;   e  = e_ts.Data;
tu = u_ts.Time;   u  = u_ts.Data;

% Common settings
LW = 2;
FS_title = 20; FS_lab = 18; FS_leg = 14; FS_ax = 16;

% Helper for y-limits: add margin +/- 10
ylims_plus10 = @(sig) [min(sig)-10, max(sig)+10];

%% ========= x1(t) =========
figure('Color','w');
set(gcf,'MenuBar','none','ToolBar','none');

plot(t1, x1, 'b', 'LineWidth', LW); grid on;
title('$x_1(t)$ evolution', 'Interpreter', 'latex', 'FontSize', FS_title);
xlabel('Time [min]', 'FontSize', FS_lab);
ylabel('$x_1$', 'Interpreter', 'latex', 'FontSize', FS_lab);
legend('$x_1(t)$','Interpreter','latex','Location','best','FontSize',FS_leg);

set(gca,'FontSize',FS_ax);
xlim([t1(1) t1(end)]);
ylim(ylims_plus10(x1));

exportgraphics(gcf, fullfile(outFolder,'x1_evolution.pdf'), 'Resolution', 300);

%% ========= x2(t) =========
figure('Color','w');
set(gcf,'MenuBar','none','ToolBar','none');

plot(t2, x2, 'b', 'LineWidth', LW); grid on;
title('$x_2(t)$ evolution', 'Interpreter', 'latex', 'FontSize', FS_title);
xlabel('Time [min]', 'FontSize', FS_lab);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', FS_lab);
legend('$x_2(t)$','Interpreter','latex','Location','best','FontSize',FS_leg);

set(gca,'FontSize',FS_ax);
xlim([t2(1) t2(end)]);
ylim(ylims_plus10(x2));

exportgraphics(gcf, fullfile(outFolder,'x2_evolution.pdf'), 'Resolution', 300);

%% ========= e(t) =========
figure('Color','w');
set(gcf,'MenuBar','none','ToolBar','none');

plot(te, e, 'b', 'LineWidth', LW); grid on;
title('e(t) evolution', 'FontSize', FS_title);
xlabel('Time [min]', 'FontSize', FS_lab);
ylabel('e(t)', 'FontSize', FS_lab);
legend('$e(t)$','Interpreter','latex','Location','best','FontSize',FS_leg);

set(gca,'FontSize',FS_ax);
xlim([te(1) te(end)]);
ylim(ylims_plus10(e));

exportgraphics(gcf, fullfile(outFolder,'e_evolution.pdf'), 'Resolution', 300);

%% ========= u(t) =========
figure('Color','w');
set(gcf,'MenuBar','none','ToolBar','none');

plot(tu, u, 'b', 'LineWidth', LW); grid on;
title('u(t) evolution', 'FontSize', FS_title);
xlabel('Time [min]', 'FontSize', FS_lab);
ylabel('u(t)', 'FontSize', FS_lab);
legend('$u(t)$','Interpreter','latex','Location','best','FontSize',FS_leg);

set(gca,'FontSize',FS_ax);
xlim([tu(1) tu(end)]);
ylim(ylims_plus10(u));

exportgraphics(gcf, fullfile(outFolder,'u_evolution.pdf'), 'Resolution', 300);
