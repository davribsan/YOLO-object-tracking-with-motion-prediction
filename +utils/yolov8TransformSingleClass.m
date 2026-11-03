function [box, cls] = yolov8TransformSingleClass(predictions)
% Converts the raw output of a YOLOv8 network into bounding box coordinates 
% and class probabilities for a single object class.
    
    % Define the maximum number of bins for Distribution Focal Loss (DFL)
    % Each bounding box coordinate (x, y, w, h) will be predicted as a 
    % distribution over regMax bins instead of a single number.
    regMax = 16; % Equal to YOLOv8

    % Compute the total number of outputs predicted per anchor
    outputsPerAnchor = 1 + regMax*4; % 1 class + box

     % Strides for small, medium, large feature maps
    stride = [8,16,32];

    % Batch size (single image)
    batch = 1;
    
    predAll = cell(3,1);
    
    % Loop over each feature map
    for i = 1:numel(predictions)

        % Rearrange dimensions
        p = permute(predictions{i}, [2,1,3,4]);

        % Flatten anchors
        p = reshape(p, [], outputsPerAnchor, batch);
         % Concatenate predictions from all scales
        predAll{i} = permute(p,[2,1,3,4]);
        
    end
    
    predCat = cat(2, predAll{:});
    
    % Split boxes and class
    box = predCat(1:64,:,:);
    cls = predCat(65,:,:); 
    
    % Anchor grid
    [anchorGrid, stride] = utils.make_anchors(predictions, stride);
    anchorGrid = anchorGrid';
    stride = stride';
    
    % Boxes
    box = utils.distributionFocalLoss(box);
    box = utils.dist2bbox(box, anchorGrid);
    box = box .* stride;
    
    % Confidence score
    cls = extractdata(sigmoid(dlarray(cls)));


end
