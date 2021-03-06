%put object containing box coordinates and other relevant information in a cell
%array
counter = 0;
for i = 1:7
    for j = 1:7
        if contain(i,j) == 1
            counter = counter+1;           
            
            % Bounding box center relative to cell
            x = outArray(i,j,22+1+(cellIndex(i,j)-1)*4);
            y = outArray(i,j,22+2+(cellIndex(i,j)-1)*4);
            
            % Yolo outputs the square root of the width and height of the
            % bounding boxes (subtle detail in paper that took me forver to realize). 
            % Relative to image size.
            w = (outArray(i,j,22+3+(cellIndex(i,j)-1)*4))^2;
            h = (outArray(i,j,22+4+(cellIndex(i,j)-1)*4))^2;
           
           %absolute values scaled to image size
            wS = w*448; 
            hS = h*448;
            xS = (j-1)*448/7+x*448/7-wS/2;
            yS = (i-1)*448/7+y*448/7-hS/2;
            
            % this array will be used for drawing bounding boxes in Matlab
            boxes(counter).coords = [xS yS wS hS]; 
            
            %save cell indices in the structure
            boxes(counter).cellIndex = [i,j];
            
            %save classIndex to structure
            boxes(counter).classIndex = classMaxIndex(i,j);    
            
            % save cell proability to structure
            boxes(counter).cellProb = cellProb(i,j);
            
            % put in a switch for non max which we will use later
            boxes(counter).nonMax = 1;
        end            
    end
end

%plot result without non-max suppression
figure(4) 
imshow(image);
hold on

for i = 1:length(boxes)
   textStr = convertStringsToChars(classLabels(boxes(i).classIndex));
   position = [(boxes(i).cellIndex(2)-1)*448/7 (boxes(i).cellIndex(1)-1)*448/7];
   text(position(1),position(2),textStr,'Color',[0 1 0],'fontWeight','bold','fontSize',12);
   rectangle('Position',boxes(i).coords, 'EdgeColor','green','LineWidth',2);
end
hold off