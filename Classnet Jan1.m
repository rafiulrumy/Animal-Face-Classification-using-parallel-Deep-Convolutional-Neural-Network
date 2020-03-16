%%Animal Face Classification
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);

imageSize = [227 227 3];
%allImages = imageDatastore('Image Dataset 227', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%[trainingImages, imdsValidation] = splitEachLabel(allImages, 0.8, 'randomize');
trainingImages = imageDatastore('Training 227', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValidation = imageDatastore('Testing 227', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
augimds = augmentedImageDatastore(imageSize,trainingImages,'DataAugmentation',imageAugmenter);
%%
layer = eluLayer('Name','elu1');
%%
layers = [...
    imageInputLayer(imageSize,'Name','data')
    convolution2dLayer(5,256,'Stride',4,'Name','conv11')
    %crossChannelNormalizationLayer(5,'Name','CN11')
    batchNormalizationLayer('Name','BN11')
    %reluLayer('Name','relu1')
    eluLayer('Name','elu11')
    maxPooling2dLayer(5,'Stride',4,'Name','pool11')
    additionLayer(2,'Name','add1')
    dropoutLayer(.25,'Name','DL11')
        
    convolution2dLayer(3,512,'Stride',2,'Name','conv21')
    %crossChannelNormalizationLayer(3,'Name','CN21')
    batchNormalizationLayer('Name','BN21')
    %reluLayer('Name','relu2')
    eluLayer('Name','elu21')
    maxPooling2dLayer(3,'Stride',1,'Name','pool21')
    additionLayer(2,'Name','add2')
    dropoutLayer(.25,'Name','DL21')
        
    fullyConnectedLayer(512,'Name','fc1')
    %reluLayer('Name','relu3')
    eluLayer('Name','elu3')
    %dropoutLayer(.25)
    
    fullyConnectedLayer(10,'Name','fc2')
    softmaxLayer('Name','prob')
    classificationLayer('Name','output')];
%%
lgraph = layerGraph(layers);
figure
plot(lgraph)
%%
Conv12 = convolution2dLayer(7,256,'Stride',4,'Padding',2,'Name','conv12');
BN12 = batchNormalizationLayer('Name','BN12');
ELU12 = eluLayer('Name','elu12');
MXPL12 = maxPooling2dLayer(13,'Stride',4,'Padding',2,'Name','pool12');
Conv22 = convolution2dLayer(5,512,'Stride',2,'Padding',2,'Name','conv22');
BN22 = batchNormalizationLayer('Name','BN22');
ELU22 = eluLayer('Name','elu22');
MXPL22 = maxPooling2dLayer(7,'Stride',4,'Padding',6,'Name','pool22');
lgraph = addLayers(lgraph,Conv12);
lgraph = addLayers(lgraph,BN12);
lgraph = addLayers(lgraph,ELU12);
lgraph = addLayers(lgraph,MXPL12);
lgraph = addLayers(lgraph,Conv22);
lgraph = addLayers(lgraph,BN22);
lgraph = addLayers(lgraph,ELU22);
lgraph = addLayers(lgraph,MXPL22);
figure
plot(lgraph)
%%
lgraph = connectLayers(lgraph,'data','conv12');
lgraph = connectLayers(lgraph,'conv12','BN12');
lgraph = connectLayers(lgraph,'BN12','elu12');
lgraph = connectLayers(lgraph,'elu12','pool12');
lgraph = connectLayers(lgraph,'pool12','add1/in2');
lgraph = connectLayers(lgraph,'DL11','conv22');
lgraph = connectLayers(lgraph,'conv22','BN22');
lgraph = connectLayers(lgraph,'BN22','elu22');
lgraph = connectLayers(lgraph,'elu22','pool22');
lgraph = connectLayers(lgraph,'pool22','add2/in2');
%%
figure
plot(lgraph);

%% Re-train the Network
options = trainingOptions('sgdm', 'InitialLearnRate',1e-4, 'MaxEpochs', 300,...
                            'Shuffle','every-epoch','ValidationData',imdsValidation,'ValidationFrequency',60,...
                            'Verbose',false,'MiniBatchSize',32,'Plots','training-progress');
%%
rng('default')
ClassNet = trainNetwork(augimds, lgraph, options);
%ClassNet = trainNetwork(augimds, layers, options);

%% Measure network accuracy

TotalAccuracy = 0;
for i = 1:10
    
    [~, imdsValidation] = splitEachLabel(allImages, 0.8, 'randomize');
    predictedLabels = classify(ClassNet, imdsValidation); 

    accuracy = mean(predictedLabels == imdsValidation.Labels)+TotalAccuracy;
    TotalAccuracy = TotalAccuracy+accuracy;
    
end
  
TotalAccuracy = TotalAccuracy/10;
%Finish

%%
%save('MyNet_2019_09_04','ClassNet');
save('ClassNet');
%%
%load('myNet.mat', 'ClassNet');
load('ClassNet');
%%
predictedLabels = classify(ClassNet, imdsValidation); 
validationImageLabels = imdsValidation.Labels;
accuracy = mean(predictedLabels == imdsValidation.Labels);
plotconfusion(validationImageLabels, predictedLabels);
%%
predictedLabels = classify(ClassNet, allImages); 
allImagesLabels = allImages.Labels;
plotconfusion(allImagesLabels, predictedLabels);
%%
X = imread('Cat.jpg');
imshow(X);
%%
X1 = imread('Human 3.jpg');
imshow(X1);
%%
X = imresize(X, [227 227]);
imshow(X);
%%
Y = classify(ClassNet,X1);
%%
Z = predict(ClassNet,X);
%% Conv11 Activation Map
channels = 1:256;
act = activations(ClassNet,X,'add1','OutputAs','channels');
act = mat2gray(act);
tmp = act(:);
tmp = imadjust(tmp,stretchlim(tmp));
act_stretched = reshape(tmp,size(act));
montage(act_stretched, 'Size', [16 16]);
title('Activations from the add1 layer','Interpreter','none');
%%