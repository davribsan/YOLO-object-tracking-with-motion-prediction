function [bboxes, scores] = detectYOLOv8_fixed(dlnet, image, executionEnvironment)
% Runs prediction on a trained yolov8 network.
%
% Inputs:
% dlnet                - Pretrained yolov8 dlnetwork.
% image                - RGB image to run prediction on. (H x W x 3)
% executionEnvironment - Environment to run predictions on. Specify cpu,
%                        gpu, or auto.
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h]) formatted as
%              NumDetections x 4.
% scores     - NumDetections x 1 classification scores.

% Copyright 2024 The MathWorks, Inc
    
    % Apply Preprocessing on the input image.
    imgSize = size(image, 1:2);
    img = single(image) * (1/255);
    
    % Convert to dlarray.
    dlInput = dlarray(img, 'SSCB');
    
    % If GPU is available, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlInput = gpuArray(dlInput);
    end
    
    % Perform prediction on the input image.
    outFeatureMaps = cell(length(dlnet.OutputNames), 1);
    [outFeatureMaps{:}] = predict(dlnet, dlInput);
    
    % Apply postprocessing on the output feature maps.
    [bboxes,scores] = utils.postprocessSingleClass_fixed(outFeatureMaps, imgSize);
    bboxes = gather(bboxes);
    scores = gather(scores);

end