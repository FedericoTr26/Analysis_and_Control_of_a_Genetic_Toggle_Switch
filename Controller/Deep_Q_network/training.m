clear; clc; close all;
rng(0,"twister");

%% --- Environment specs ---
maxConc = 10;
observationSpec = rlNumericSpec([2 1], ...
    "LowerLimit",[0;0], ...
    "UpperLimit",[maxConc; maxConc]);
observationSpec.Name = "Concentrations";


numActions = 7;
actionSpec = rlFiniteSetSpec(1:numActions);
actionSpec.Name = "InputIndex";

% Crea l'ambiente chiamando le funzioni negli altri file
env = rlFunctionEnv(observationSpec, actionSpec, "toggleStep_DQN", "toggleReset_DQN");


sampleTime = 0.5;  % [s] MUST match dt in toggleStep_DQN
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Q-network structure
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

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.005;
agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.01;

agent = rlDQNAgent(critic, agentOpts);


steps15 = round((15*60)/sampleTime);   % 15 minutes = 1800 steps

%% --- Training options ---
trainOpts = rlTrainingOptions( ...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=steps15, ...          
    ScoreAveragingWindowLength=20, ...
    StopTrainingCriteria="AverageReward", ...
    StopTrainingValue=400, ...               
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

% Map action indices to u values
actionValues = linspace(-10,30,numActions);
u_ctrl = actionValues(actionIdx);                      

% Time axes (minutes)
t_x = (0:numel(x1)-1)' * sampleTime / 60;              
t_u = (0:numel(u_ctrl)-1)' * sampleTime / 60;          

% Error computed on state samples
xS = [1.92335; 1.92335];
err = sqrt((x1 - xS(1)).^2 + (x2 - xS(2)).^2);

%% --- Plots ---
% 1. Phase Plane
figure('Color','w','Name','Phase plane trajectory'); hold on; grid on; axis square;
plot(x1, x2,'k-','LineWidth',1.5,'DisplayName','DQN trajectory');
plot(0.12702, 7.87298,'bo','MarkerFaceColor','b','DisplayName','xB (start)');
plot(1.92335, 1.92335,'rx','LineWidth',2,'MarkerSize',10,'DisplayName','xS (target)');
plot(7.87298, 0.12702,'ko','MarkerFaceColor','k','DisplayName','xA (stable)');
xlabel('Concentration x_1'); ylabel('Concentration x_2');
title('Trajectory in Phase Plane');
legend('Location','best');

% 2. State Evolution (Time)
figure('Color','w','Name','State evolution'); hold on; grid on;
plot(t_x, x1,'LineWidth',2,'DisplayName','x_1');
plot(t_x, x2,'LineWidth',2,'DisplayName','x_2');
yline(xS(1),'--k','DisplayName','x_S level');
xlabel('Time [min]'); ylabel('Concentration');
title('State Trajectories');
xlim([0 15]); 
legend('Location','best');

% 3. Performance
figure('Color','w','Name','Performance analysis');
subplot(2,1,1);
plot(t_x, err,'k','LineWidth',1.5); grid on;
xlabel('Time [min]'); ylabel('||x-x_S||');
title('Error (distance from saddle)');
xlim([0 15]);
subplot(2,1,2);
stairs(t_u, u_ctrl,'LineWidth',1.2); grid on;
xlabel('Time [min]'); ylabel('u');
title('Control input');
xlim([0 15]);