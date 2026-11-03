classdef YOLOv8
    properties
        detector      % Loaded model
        classNames    % Classes names
        numClasses    % Number of classes
        inputShape    % Expected image shape [h, w]
    end
    
    methods
        % Constructor of Inference class
        function obj = YOLOv8(modelPath, inputShape)
            obj.detector = importYOLOv8Model(modelPath); % Load ONNX model
            % Initialize dlnetwork
            obj.detector = initialize(obj.detector);    
            obj.inputShape = inputShape;
        end
        
        % Perform inference on images
        function [bboxes, scores, labels] = runInference(obj, I)
            
            % Perform detection using pretrained model
            [bboxes, scores] = detectYOLOv8(obj.detector, I, 'auto');
            
            % Map labelIds back to labels
            labels = repmat("UAV", size(bboxes,1), 1);              

        end      
    end
end
