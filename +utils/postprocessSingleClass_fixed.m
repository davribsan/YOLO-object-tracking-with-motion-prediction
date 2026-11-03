function [bboxes, scores] = postprocessSingleClass_fixed(outFeatureMaps, imgSize)
% function [bboxes, scores] = postprocessSingleClass_fixed(outFeatureMaps, imgSize)
% Takes the raw feature maps produced by a YOLOv8 network and converts them
% into final bounding boxes and confidence scores for a single object class.

    % Transform output: box [x y w h] and score
    [box, score] = utils.yolov8TransformSingleClass(outFeatureMaps);
    
    % Concatenate predictions into a single array
    predictions = cat(1, box, score);
    
    % Extract valid detections for the single class
    [bboxesPred, scores] = utils.extractDetectionsSingleClass(predictions);

    %bboxes = bboxesPred;
    
    % Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2]
    bboxesTmp = xywhToX1Y1X2Y2(bboxesPred);
    
    % Rescale bounding boxes from resized input back to original image size
    bboxesPost = utils.postProcessYOLOv8_fixed(imgSize, bboxesTmp);
    
    % Convert bounding boxes back to [x, y, w, h] format
    bboxes = x1y1x2y2ToXYWH(bboxesPost);


end


function boxes = xywhToX1Y1X2Y2(boxes)
    boxes(:,3) = boxes(:,1) + boxes(:,3) - 1;
    boxes(:,4) = boxes(:,2) + boxes(:,4) - 1;
end

function boxes = x1y1x2y2ToXYWH(boxes)
    boxes(:,3) = boxes(:,3) - boxes(:,1) + 1;
    boxes(:,4) = boxes(:,4) - boxes(:,2) + 1;
end

