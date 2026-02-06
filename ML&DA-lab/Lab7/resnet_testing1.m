%unzip('MerchData.zip');
%load resnet_hcc3.mat;
%net=resnet101;
%unzip('C:\Users\dmitrea\Documents\delia_new1.zip');
%imds = imageDatastore('MerchData','IncludeSubfolders',true, 'LabelSource','foldernames'); 
imds = imageDatastore('TumPancreatice1','IncludeSubfolders',true, 'LabelSource','foldernames'); 
%[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds,0.75, 0.08, 0.17);
net=resnet101;
%load resnet101d2_hcc3.mat;
%net=resnet101d2_hcc4;
%net=trainedNetwork_1_ResNet50;
inputSize = net.Layers(1).InputSize;
 if isa(net,'SeriesNetwork') 
     lgraph = layerGraph(net.Layers); 
 else
   lgraph = layerGraph(net);
 end
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
%[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
newLearnableLayer = convolution2dLayer(1,numClasses, 'Name','new_conv', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

%freeze!!!

layers = lgraph.Layers;
connections = lgraph.Connections;

%layers(1:5) = freezeWeights(layers(1:5));
%lgraph = createLgraphUsingConnections(layers,connections);

%end freeze!!!

%image augmentation

pixelRange = [-150 150];
scaleRange = [0.9 1.1];
rotRange=[-180 180];

imageAugmenter = imageDataAugmenter('RandXReflection',true, 'RandXTranslation',pixelRange, 'RandYTranslation',pixelRange, 'RandXScale',scaleRange, 'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

miniBatchSize = 30;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm','MiniBatchSize',miniBatchSize, 'MaxEpochs',70, 'InitialLearnRate',e-4, 'Shuffle','every-epoch', 'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false,...
    'ExecutionEnvironment', 'gpu',...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);
[YPred,probs] = classify(net,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);
plotconfusion(imdsTest.Labels,YPred);




