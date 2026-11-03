function [bboxes, scores] = detectYOLOv8(dlnet, image, executionEnvironment)
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

    % Get the input size of the network.
    inputSize = dlnet.Layers(1).InputSize;
    
    % Apply Preprocessing on the input image.
    origSize = size(image);
    %Ibgr = image(:,:,[3,2,1]); % convert image to bgr
    
    %img = utils.preprocess(Ibgr, inputSize);
    img = utils.preprocess(image, inputSize);
    
    newSize = size(img);
    %img = img(:,:,[3,2,1]); % convert image to rgb
    
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
    [bboxes,scores] = utils.postprocessSingleClass(outFeatureMaps, ...
         origSize, newSize);
    bboxes = gather(bboxes);
    scores = gather(scores);

end