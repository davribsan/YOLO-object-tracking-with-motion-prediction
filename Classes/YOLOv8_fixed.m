classdef YOLOv8_fixed
    properties
        detector      % Loaded model
        classNames    % Classes names
        numClasses    % Number of classes
        inputShape    % Expected image shape [h, w]
    end
    
    methods
        % Constructor of Inference class
        function obj = YOLOv8_fixed(modelPath, inputShape)
            obj.detector = importYOLOv8Model(modelPath); % Load ONNX model
            % Initialize dlnetwork
            obj.detector = initialize(obj.detector);    
            obj.inputShape = inputShape;
        end
        
        % Perform inference on images
        function [bboxes, scores, labels] = runInference_fixed(obj, I)
            
            % Perform detection using pretrained model
            [bboxes, scores] = detectYOLOv8_fixed(obj.detector, I, 'auto');
            
            % Map labelIds back to labels
            labels = repmat("UAV", size(bboxes,1), 1);              

        end      
    end
end
