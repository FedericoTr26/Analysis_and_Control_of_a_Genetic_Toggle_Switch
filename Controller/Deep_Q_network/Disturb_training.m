clear; clc; close all;
rng(0,"twister");

maxConc = 10;
observationSpec = rlNumericSpec([2 1], ...
    "LowerLimit",[0;0], ...
    "UpperLimit",[maxConc; maxConc]);
observationSpec.Name = "Concentrations";

numActions = 7;
actionSpec = rlFiniteSetSpec(1:numActions);
actionSpec.Name = "InputIndex";


env = rlFunctionEnv(observationSpec, actionSpec, "Disturb_toggleStep_DQN", "Disturb_toggleReset_DQN");

sampleTime = 0.5;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

layers = [
    featureInputLayer(prod(obsInfo.Dimension), Name="Input")
    fullyConnectedLayer(64, Name="FC1")
    reluLayer(Name="ReLU1")
    fullyConnectedLayer(64, Name="FC2")
    reluLayer(Name="ReLU2")
    fullyConnectedLayer(numel(actInfo.Elements), Name="Qout")
];
dlnet  = dlnetwork(layers);
critic = rlVectorQValueFunction(dlnet, obsInfo, actInfo);

optimizerOpts = rlOptimizerOptions(LearnRate=1e-3, GradientThreshold=1);
agentOpts = rlDQNAgentOptions( ...
    SampleTime=sampleTime, ...
    CriticOptimizerOptions=optimizerOpts, ...
    ExperienceBufferLength=50000, ...
    DiscountFactor=0.99, ...
    MiniBatchSize=128);

% Exploration decay (un po' più lento dato il rumore)
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.001; 
agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.01;

agent = rlDQNAgent(critic, agentOpts);


steps15 = round((15*60)/sampleTime);


trainOpts = rlTrainingOptions( ...
    MaxEpisodes=1500, ...                    
    MaxStepsPerEpisode=steps15, ...          
    ScoreAveragingWindowLength=20, ...
    StopTrainingCriteria="AverageReward", ...
    StopTrainingValue=350, ...               
    Verbose=true, ...
    Plot="training-progress");

doTraining = true;
if doTraining
    trainingStats = train(agent, env, trainOpts);
end


simOpts = rlSimulationOptions(MaxSteps=steps15);
simResult = sim(env, agent, simOpts);


obsData = simResult.Observation.Concentrations.Data;   
x1 = squeeze(obsData(1,1,:));   
x2 = squeeze(obsData(2,1,:));   
actData = simResult.Action.InputIndex.Data;            
actionIdx = squeeze(actData);                          
actionIdx = actionIdx(:);

actionValues = linspace(-10,30,numActions);
u_ctrl = actionValues(actionIdx);                      

t_x = (0:numel(x1)-1)' * sampleTime / 60;              
t_u = (0:numel(u_ctrl)-1)' * sampleTime / 60;          

xS = [1.92335; 1.92335];
err = sqrt((x1 - xS(1)).^2 + (x2 - xS(2)).^2);


% Definisco i font size grandi per il report
fs_title = 24;
fs_axis  = 22;
fs_ticks = 16;
fs_leg   = 16;

% ---------------------------------------------------------
% FIGURA 1: Phase Plane (Piano delle Fasi)
% ---------------------------------------------------------
figure('Color','w','Name','1. Phase Plane', 'Position', [100 100 800 600]);
plot(x1, x2,'k-','LineWidth',1.5, 'DisplayName','DQN Trajectory'); hold on;
plot(0.12702, 7.87298,'bo','MarkerFaceColor','b','MarkerSize',10, 'DisplayName','x_B (Start)');
plot(1.92335, 1.92335,'rx','LineWidth',3,'MarkerSize',12, 'DisplayName','x_S (Target)');

xlabel('Concentration x_1', 'FontSize', fs_axis); 
ylabel('Concentration x_2', 'FontSize', fs_axis); 
title('Phase Plane Trajectory', 'FontSize', fs_title);

legend('Location','best', 'FontSize', fs_leg);
grid on; axis square;
set(gca, 'FontSize', fs_ticks);

% ---------------------------------------------------------
% FIGURA 2: State Evolution (Traiettorie nel Tempo)
% ---------------------------------------------------------
figure('Color','w','Name','2. State Evolution', 'Position', [150 150 800 600]);
plot(t_x, x1, 'LineWidth', 2, 'DisplayName', 'x_1'); hold on;
plot(t_x, x2, 'LineWidth', 2, 'DisplayName', 'x_2');
yline(xS(1), '--k', 'LineWidth', 1.5, 'DisplayName', 'Target Level (x_S)');

xlabel('Time [min]', 'FontSize', fs_axis); 
ylabel('Concentration', 'FontSize', fs_axis); 
title('State Evolution', 'FontSize', fs_title);

legend('Location','best', 'FontSize', fs_leg);
xlim([0 15]);
grid on;
set(gca, 'FontSize', fs_ticks);

% ---------------------------------------------------------
% FIGURA 3: Error (Distanza dal Saddle)
% ---------------------------------------------------------
figure('Color','w','Name','3. Error Analysis', 'Position', [200 200 800 600]);
plot(t_x, err,'r','LineWidth',1.5); 
grid on;

% Qui ho rimesso i nomi "classici" per evitare confusione
title('Error', 'FontSize', fs_title);
xlabel('Time [min]', 'FontSize', fs_axis); 
ylabel('||x - x_S||', 'FontSize', fs_axis);

xlim([0 15]);
set(gca, 'FontSize', fs_ticks);

% ---------------------------------------------------------
% FIGURA 4: Control Input (Ingresso)
% ---------------------------------------------------------
figure('Color','w','Name','4. Control Input', 'Position', [250 250 800 600]);
stairs(t_u, u_ctrl,'b','LineWidth',1.5); 
grid on;

title('Control Input', 'FontSize', fs_title);
xlabel('Time [min]', 'FontSize', fs_axis); 
ylabel('Input u', 'FontSize', fs_axis);

xlim([0 15]);
set(gca, 'FontSize', fs_ticks);