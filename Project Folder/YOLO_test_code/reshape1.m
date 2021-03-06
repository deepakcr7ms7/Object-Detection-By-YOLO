% in order to make interpretation of this output vector more manageable
% with my finite visual-spatial skills, I decided to reshape the vector
% into a 7x7x30 array with the third dimension containing all information
% for each of the 7x7 cells.

class = out(1:980);
boxProbs = out(981:1078);
boxDims = out(1079:1470);

outArray = zeros(7,7,30);
temp = class;
    for j = 0:6
        for i = 0:6
            outArray(i+1,j+1,1:20) = class(i*20*7+j*20+1:i*20*7+j*20+20);
            outArray(i+1,j+1,21:22) = boxProbs(i*2*7+j*2+1:i*2*7+j*2+2);
            outArray(i+1,j+1,23:30) = boxDims(i*8*7+j*8+1:i*8*7+j*8+8);
        end
    end

%find boxes with probabilities above a defined probability threshold. 0.2 seems to
%work well. cellIndex tells us which of two bounding boxes for each cell
%has higher probability. 1 for vertical box and 2 for horizontal box.

[cellProb cellIndex] = max(outArray(:,:,21:22),[],3);
contain = max(outArray(:,:,21:22),[],3)>probThresh;

%find highest probability object class type for each cell, save it to
%classMaxIndex
[classMax,classMaxIndex] = max(outArray(:,:,1:20),[],3);

figure(3)
imagesc(contain);