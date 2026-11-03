function [bboxes,scores,labels] = yoloTrackingWorker(frame, modelPath, inputShape, modelType)
    persistent yoloDetection yoloTracking

    if strcmp(modelType, 'detection') && isempty(yoloDetection)
        disp('Initializing detection model...')
        yoloDetection = YOLOv8(modelPath, inputShape);
    elseif strcmp(modelType, 'tracking') && isempty(yoloTracking)
        disp('Initializing tracking model...')
        yoloTracking = YOLOv8_fixed(modelPath, inputShape);
    end

    if nargin == 0 || isempty(frame)
        bboxes = []; scores = []; labels = [];
        return
    end

    if strcmp(modelType, 'tracking')
        [bboxes, scores, labels] = yoloTracking.runInference_fixed(frame);
    else
        [bboxes, scores, labels] = yoloDetection.runInference(frame);
    end
end