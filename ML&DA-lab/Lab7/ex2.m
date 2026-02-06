% Minimal InceptionV3 Training on CalTech101
clc; clear; close all;

datasetPath = '/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/caltech-101/101/caltech101';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');


%% Split dataset: 70% train / 30% test

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');


%% Load pre-trained InceptionV3

net = inceptionv3;
inputSize = net.Layers(1).InputSize;  % typically [299 299 3]
numClasses = numel(categories(imdsTrain.Labels));


%% Replace final layers for transfer learning

lgraph = layerGraph(net);

% Fully connected layer
newFc = fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10, 'Name', 'new_fc');

lgraph = replaceLayer(lgraph, 'predictions', newFc);

%Classification layer
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

%% Image Augmentation & Resize

augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');


%% Training options

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 3, ...          % set mic pentru test rapid
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);


%% Train the network

netTransfer = trainNetwork(augImdsTrain, lgraph, options);


%% Evaluate on test set

YPred = classify(netTransfer, augImdsTest);
accuracy = mean(YPred == imdsTest.Labels);

disp(['InceptionV3 Accuracy: ', num2str(accuracy*100), '%']);

%% plot

% Predicții pe test set
YPred = classify(netTransfer, augImdsTest);
YTest = imdsTest.Labels;

% Confusion matrix
figure;
confMat = confusionmat(YTest, YPred);
confMatChart = confusionchart(confMat, categories(YTest));
confMatChart.Title = 'Confusion Matrix';
confMatChart.RowSummary = 'row-normalized';
confMatChart.ColumnSummary = 'column-normalized';

