function [bboxes,scores] = extractDetectionsSingleClass(predictions)
% Processes the raw predictions for a single object class and returns the 
% filtered bounding boxes and confidence scores after applying a confidence 
% threshold and non-maximum suppression (NMS).

    confThreshold = 0.45;
    iouThreshold  = 0.45;
    
    % Predictions = [x y w h score]
    xc = predictions(5,:) > confThreshold;
    
    if ~any(xc)
        bboxes = zeros(0,4);
        scores = zeros(0,1);
        return
    end
    
    pred = predictions(:, xc)';
    
    box   = pred(:,1:4);
    scores = pred(:,5);
    
    % Convert center → top-left
    box(:,1:2) = floor(box(:,1:2) - box(:,3:4)/2 + 0.5);
    box(:,1:2) = max(box(:,1:2),1);
    
    % NMS (una sola clase)
    [bboxes, scores] = selectStrongestBbox( ...
        box, scores, ...
        'RatioType','Union', ...
        'OverlapThreshold', iouThreshold);


end
