
% Minimal ResNet50 on CalTech101


datasetPath = "/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/date/caltech-101/101/caltech101";

%% Load images
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');


%% Split 70% train / 30% test
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

%% Augmented datastores with resizing + grayscale->RGB
inputSize = [224 224]; % ResNet50 expects 224x224
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing','gray2rgb');
augImdsTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','gray2rgb');

%% Load pretrained ResNet50
net = resnet50;
numClasses = numel(categories(imdsTrain.Labels));

%% Replace final layers
layers = layerGraph(net);
layers = replaceLayer(layers, 'fc1000', ...
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10));
layers = replaceLayer(layers, 'ClassificationLayer_fc1000', ...
    classificationLayer);

%% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',2, ...           
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',true);

%% Train the network
netTransfer = trainNetwork(augImdsTrain, layers, options);

%% Test accuracy
YPred = classify(netTransfer, augImdsTest);
accuracy = mean(YPred == imdsTest.Labels);
disp(['ResNet50 Accuracy: ', num2str(accuracy*100), '%']);
