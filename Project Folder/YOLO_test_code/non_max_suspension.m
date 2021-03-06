% Begin non max suppression. If intersection over union of any two bounding boxes is 
% higher than a defined threshold, and the two boxes contain the same class,
% remove the box with the the lower box probability.

for i = 1:length(boxes)
    for j = i+1:length(boxes)
        %calculate intersection over union (can also use bboxOverlapRatio
        %with proper toolbox
        intersect = rectint(boxes(i).coords,boxes(j).coords);
        union = boxes(i).coords(3)*boxes(i).coords(4)+boxes(j).coords(3)*boxes(j).coords(4)-intersect;
        iou(i,j) = intersect/union;
        if boxes(i).classIndex == boxes(j).classIndex && iou(i,j) > iouThresh                
            [value(i) dropIndex(i)] = min([boxes(i).cellProb boxes(j).cellProb]);
            if dropIndex(i) == 1
                boxes(i).nonMax=0;
            elseif dropIndex(i) == 2
                boxes(j).nonMax=0;                
            end
        end                
    end
end

%plot result with non max suppression
figure(5) 
imshow(image);
hold on
for i = 1:length(boxes)
    if boxes(i).nonMax == 1
   textStr = convertStringsToChars(classLabels(boxes(i).classIndex));
   position = [(boxes(i).cellIndex(2)-1)*448/7 (boxes(i).cellIndex(1)-1)*448/7];
   text(position(1),position(2),textStr,'Color',[0 1 0],'fontWeight','bold','fontSize',12);
   rectangle('Position',boxes(i).coords, 'EdgeColor','green','LineWidth',2);
    end
end