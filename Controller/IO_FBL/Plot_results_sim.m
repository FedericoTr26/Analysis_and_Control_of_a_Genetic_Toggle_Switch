clear; clc; close all;

model = "IO_FBL_control";

% Output folder
outDir = fullfile(pwd, "plots_pdf");
if ~exist(outDir, "dir")
    mkdir(outDir);
end
fprintf("Saving PDFs to:\n  %s\n\n", outDir);

% Readable text for export
set(groot, ...
    "defaultFigureColor", "w", ...
    "defaultAxesFontSize", 16, ...
    "defaultTextFontSize", 16, ...
    "defaultAxesLineWidth", 1.2, ...
    "defaultLineLineWidth", 2.0, ...
    "defaultTextInterpreter", "latex", ...
    "defaultAxesTickLabelInterpreter", "latex", ...
    "defaultLegendInterpreter", "latex");

% Run simulation
load_system(model);
simOut = sim(model, "ReturnWorkspaceOutputs", "on");

% Get timeseries (from simOut or base workspace)
x1_ts = localGetTimeseries(simOut, "x1");
x2_ts = localGetTimeseries(simOut, "x2");
u_ts  = localGetTimeseries(simOut, "u");
e_ts  = localGetTimeseries(simOut, "error");

% Plot + save (figures remain visible)
localPlotSavePDF_visible(x1_ts, "$x_1(t)$ evolution", "$x_1$", "$x_1(t)$",        fullfile(outDir, "x1_evolution.pdf"));
localPlotSavePDF_visible(x2_ts, "$x_2(t)$ evolution", "$x_2$", "$x_2(t)$",        fullfile(outDir, "x2_evolution.pdf"));
localPlotSavePDF_visible(e_ts,  "$e(t)$ evolution",   "$e$",   "Error",           fullfile(outDir, "e_evolution.pdf"));
localPlotSavePDF_visible(u_ts,  "$u(t)$ evolution",   "$u$",   "Control input",   fullfile(outDir, "u_evolution.pdf"));

disp("Done. Figures are open and PDFs have been saved.");

%% ===== Local functions =====
function ts = localGetTimeseries(simOut, varName)
    ts = [];
    try
        ts = simOut.get(varName);
    catch
    end
    if isempty(ts)
        try
            ts = evalin("base", varName);
        catch
            error("Cannot find '%s' in simOut or base workspace. Check To Workspace variable names.", varName);
        end
    end
    if ~isa(ts, "timeseries")
        error("'%s' is not a timeseries. Set To Workspace -> Save format = timeseries.", varName);
    end
end

function localPlotSavePDF_visible(ts, ttl, ylab, legStr, pdfPath)
    t = ts.Time(:);       
    y = squeeze(ts.Data);

    if ~isvector(y)
        error("Signal for '%s' is not 1-D. Data size is %s.", ttl, mat2str(size(ts.Data)));
    end

    fig = figure("Visible","on", "Units","pixels", "Position",[120 120 980 560]);
    set(fig, "Color","w", "Renderer","painters");

    ax = axes(fig); %#ok<LAXES>
    plot(ax, t, y, "b");
    grid(ax, "on");

    % ---- Bigger, less "thin" text ----
    ax.FontSize = 18;              % ticks size
    ax.LineWidth = 1.2;

    % If you want a heavier font (optional):
    % ax.FontName = "Times New Roman";

    xlabel(ax, "Time [min]", "Interpreter","latex", "FontSize", 20, "FontWeight","normal");
    ylabel(ax, ylab,         "Interpreter","latex", "FontSize", 20, "FontWeight","normal");

    % Bold title + larger
    title(ax, ttl, "Interpreter","latex", "FontSize", 22, "FontWeight","bold");

    lgd = legend(ax, legStr, "Location","best");
    lgd.Box = "on";
    lgd.Interpreter = "latex";
    lgd.FontSize = 18;
    % lgd.FontName = "Times New Roman"; % optional heavier font

    % Tight layout
    set(ax, "LooseInset", max(get(ax,"TightInset"), 0.02));

    % Export tightly cropped vector PDF
    exportgraphics(fig, pdfPath, "ContentType","vector", "BackgroundColor","white");

    if exist(pdfPath, "file")
        fprintf("Saved: %s\n", pdfPath);
    else
        warning("FAILED to save: %s", pdfPath);
    end
end
