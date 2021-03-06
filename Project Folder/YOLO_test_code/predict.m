probThresh = 0.10;
iouThresh = 0.4;   

%preprocess an image for analysis. For my image of a dog named Stella, I need 
% to resize to 448x448 pixels, and convert from an
%unsigned 8bit to a single with pixel values scaled to [0,1]. I could not find
%the appropriate pixel value scale for yolo recorded anywhere, but convinced
%myself through reverse engineering that [0,1] is correct.

image = single(imresize(imread('download.jpg'),[448 448]))/255;
figure(1);
imagesc(image);

% Define 20 class labels that yolo has been trained on. Classes are in
% alphabetical order.
classLabels = ["aeroplane",	"bicycle",	"bird"	,"boat",	"bottle"	,"bus"	,"car",...
"cat",	"chair"	,"cow"	,"diningtable"	,"dog"	,"horse",	"motorbike",	"person",	"pottedplant",...
"sheep",	"sofa",	"train",	"tvmonitor"];

%run the image through the network. Replace 'gpu' with 'cpu' if you do not
%have a CUDA enbled GPU.

tic
out = predict(yoloml,image,'ExecutionEnvironment','cpu');
toc

%plot the 1x1470 output vector. Indices 1-980 are class probabilities,
%981-1079 are cell/box probabilities, and 1080-1470 are bounding box parameters
figure(2)
plot(out)