%% 1. Inputs
% Include the project folder and all subfolders to the MATLAB's path
addpath(genpath(pwd));

% Path to the video
videoPath = fullfile(pwd, 'videos', 'video18.mp4');   % put your video in the videos/ folder

% Path to the neural network model for first detection or after missing objects 
modelPathDetection = fullfile(pwd, 'Models', 'best_640_8.onnx');

% Path to the neural network model for tracking
modelPathTracking = fullfile(pwd, 'Models', 'best_320_8.onnx');                

% Size of the image that the neural network will process for first detection or after missing objects 
imageSizeDetection = 640;  

% Size of the image that the neural network will process for tracking
imageSizeTracking = 320;


%% 2. Model Initialization

% Size of the square images that the neural network will process
inputShapeDetection = [imageSizeDetection, imageSizeDetection]; 
inputShapeTracking = [imageSizeTracking, imageSizeTracking]; 

% Initialize the parallel pool required to run YOLO inference in background workers
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local',2); 
end

% Initialize the heavier model in the first worker
futureInit1 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathDetection, inputShapeDetection, 'detection');
fetchOutputs(futureInit1);
fprintf('\nDetection model ready\n')

% initialize the lighter model in the second worker
futureInit2 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathTracking, inputShapeTracking, 'tracking');
fetchOutputs(futureInit2);
fprintf('Tracking model ready \n\n')


 %% 3. Tracking

% Video setup
video  = VideoReader(videoPath);
player = vision.VideoPlayer;
fps    = video.FrameRate;
dt     = 1/fps;                 % time between frames [s]

% Kalman filter state
kf            = [];             % trackingKF object
kfInitialized = false;
adaptiveQ     = 200;            % process noise (adapted to target speed)
adaptiveR     = 10;             % measurement noise (adapted to target speed)
lastCenter    = [];             % last measured center [x, y], used for speed estimation

% Tracking state machine
isTracking      = false;        % true once a target was detected
isConfirming    = false;        % true while the detection is being validated
confirmCount    = 0;            % consecutive confirmations so far
minConfirmCount = 2;            % confirmations needed to start tracking
missCount       = 0;            % consecutive frames without detection
maxMisses       = 5;            % misses allowed before going back to detection

% Bounding boxes
bboxes       = [];              % bbox drawn in the current frame [x y w h]
lastBboxes   = [];              % last bbox returned by the model (size reference)
lastPredBbox = [];              % last Kalman-predicted bbox while searching

% Prediction while the detector is searching
predFrameCount = 0;             % frames predicted by Kalman only
maxPredFrames  = 10;            % after this, the bbox is frozen in place

% Asynchronous inference
futureInference = [];           % parfeval future (empty = no task running)
% Crop info needed to map tracking-model output back to full-frame coordinates
sentX1 = 1; sentY1 = 1; sentScaleX = 1; sentScaleY = 1;

frameNumber = 0;

% Main loop
while hasFrame(video)
    tic
    frame = readFrame(video);
    frameNumber = frameNumber + 1;

    % 1 - Kalman prediction in intermediate frames (tracking mode only)
    % While the model is still computing, move the bbox with the Kalman prediction.
    if isTracking && ~isConfirming && kfInitialized && ~isempty(lastBboxes) && ~isempty(futureInference)
        kfPred = predict(kf);
        w = lastBboxes(3);
        h = lastBboxes(4);
        bboxes = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];   % center -> top-left corner
    end

    % 2 - Kalman prediction while the detection model is searching
    % The target was lost: keep extrapolating for a limited number of frames,
    % then freeze the bbox at the last predicted position.
    if ~isTracking && kfInitialized && ~isempty(lastBboxes) && ~isempty(futureInference)
        if predFrameCount < maxPredFrames
            kfPred = predict(kf);
            w = lastBboxes(3);
            h = lastBboxes(4);
            bboxes         = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];
            lastPredBbox   = bboxes;
            predFrameCount = predFrameCount + 1;
        else
            bboxes = lastPredBbox;   % frozen
        end
    end

    % 3 - Launch inference if no task is active
    if isempty(futureInference)
        if isTracking && ~isempty(lastBboxes)
            % ---- Tracking mode: crop centered on the target ----
            if kfInitialized && ~isConfirming
                % Use the Kalman state as crop center, but limit how far it
                % can drift from the last measured position.
                state = kf.State;
                xc = state(1);
                yc = state(3);

                xcLast   = lastBboxes(1) + lastBboxes(3)/2;
                ycLast   = lastBboxes(2) + lastBboxes(4)/2;
                maxDrift = imageSizeTracking * 0.5;
                xc = xcLast + max(-maxDrift, min(maxDrift, xc - xcLast));
                yc = ycLast + max(-maxDrift, min(maxDrift, yc - ycLast));
            else
                % Not confirmed yet: center the crop on the last detection
                xc = lastBboxes(1) + lastBboxes(3)/2;
                yc = lastBboxes(2) + lastBboxes(4)/2;
            end

            % Crop limits, clamped to the frame borders
            x1 = round(xc - imageSizeTracking/2);
            y1 = round(yc - imageSizeTracking/2);
            x2 = x1 + imageSizeTracking - 1;
            y2 = y1 + imageSizeTracking - 1;

            x1 = max(1, x1); y1 = max(1, y1);
            x2 = min(x2, size(frame,2)); y2 = min(y2, size(frame,1));

            cropRaw = frame(y1:y2, x1:x2, :);
            cropH = y2 - y1 + 1;
            cropW = x2 - x1 + 1;

            % If the crop was clipped at a border, resize it to the model
            % input size and remember the scale to undo it later.
            if cropH ~= imageSizeTracking || cropW ~= imageSizeTracking
                crop   = imresize(cropRaw, [imageSizeTracking, imageSizeTracking]);
                scaleX = cropW / imageSizeTracking;
                scaleY = cropH / imageSizeTracking;
            else
                crop   = cropRaw;
                scaleX = 1; scaleY = 1;
            end

            % Store crop offset/scale to map the output back to the full frame
            sentX1 = x1; sentY1 = y1;
            sentScaleX = scaleX; sentScaleY = scaleY;

            futureInference = parfeval(pool, @yoloTrackingWorker, 3, ...
                crop, modelPathTracking, inputShapeTracking, 'tracking');
        else
            % ---- Detection mode: full frame ----
            futureInference = parfeval(pool, @yoloTrackingWorker, 3, ...
                frame, modelPathDetection, inputShapeDetection, 'detection');
        end
    end

    % 4 - Get results when the inference has finished
    if ~isempty(futureInference) && strcmp(futureInference.State, 'finished')
        try
            [bboxesTmp, ~, ~] = fetchOutputs(futureInference);
        catch
            bboxesTmp = [];   % worker failed: treat as "no detection"
        end
        futureInference = [];

        if isTracking
            % ===================== TRACKING MODE =====================
            if ~isempty(bboxesTmp)
                % Transform crop coordinates back to full-frame coordinates
                bboxes      = bboxesTmp;
                bboxes(:,1) = bboxesTmp(:,1) * sentScaleX + sentX1 - 1;
                bboxes(:,2) = bboxesTmp(:,2) * sentScaleY + sentY1 - 1;
                bboxes(:,3) = bboxesTmp(:,3) * sentScaleX;
                bboxes(:,4) = bboxesTmp(:,4) * sentScaleY;
                lastBboxes  = bboxes(1,:);

                % Center of the measured bbox
                xc = lastBboxes(1) + lastBboxes(3)/2;
                yc = lastBboxes(2) + lastBboxes(4)/2;

                if isConfirming
                    % --- Confirmation phase ---
                    confirmCount = confirmCount + 1;

                    if confirmCount >= minConfirmCount
                        disp('Target confirmed. Starting tracking...')
                        isConfirming = false;

                        % Create the Kalman filter (2D constant velocity)
                        % State = [x; vx; y; vy]
                        kf = trackingKF('MotionModel', '2D Constant Velocity', ...
                                        'State',            [xc;0;yc;0], ...
                                        'MeasurementNoise', adaptiveR, ...
                                        'ProcessNoise',     adaptiveQ);
                        kfInitialized = true;
                        lastCenter    = [xc, yc];
                    end
                else
                    % --- Normal tracking phase ---
                    if ~kfInitialized
                        kf = trackingKF('MotionModel', '2D Constant Velocity', ...
                                        'State',            [xc;0;yc;0], ...
                                        'MeasurementNoise', adaptiveR, ...
                                        'ProcessNoise',     adaptiveQ);
                        kfInitialized = true;
                        lastCenter    = [xc, yc];
                    else
                        % Adapt noise parameters to the estimated speed
                        if ~isempty(lastCenter)
                            speed = norm([xc, yc] - lastCenter) / dt;
                            if speed > 70
                                adaptiveQ = 300;
                                adaptiveR = 5;    % fast target: trust YOLO more
                            elseif speed > 15
                                adaptiveQ = 200;
                                adaptiveR = 10;   % normal speed
                            else
                                adaptiveQ = 50;
                                adaptiveR = 20;   % slow target: smooth more
                            end
                            kf.ProcessNoise     = adaptiveQ;
                            kf.MeasurementNoise = adaptiveR;
                        end
                        lastCenter = [xc, yc];

                        % Correct the filter with the new measurement
                        correct(kf, [xc, yc]);
                    end
                end

                % Target seen: reset the loss counters
                missCount      = 0;
                predFrameCount = 0;
                lastPredBbox   = [];

                % Display the filtered position (smoother than the raw detection)
                if ~isConfirming
                    state  = kf.State;
                    bboxes = [state(1) - lastBboxes(3)/2, state(3) - lastBboxes(4)/2, ...
                              lastBboxes(3), lastBboxes(4)];
                end

            else
                % No detection in this crop
                missCount = missCount + 1;

                if isConfirming
                    % Confirmation failed: go back to full-frame detection
                    disp('Confirmation failed. Back to detection...')
                    isTracking   = false;
                    isConfirming = false;
                    confirmCount = 0;
                    missCount    = 0;
                    lastBboxes   = [];
                    bboxes       = [];
                    lastCenter   = [];
                else
                    disp(['Target not detected. Attempt ' num2str(missCount) '/' num2str(maxMisses)])
                    if missCount >= maxMisses
                        % Too many misses: target lost, back to detection.
                        % The Kalman filter is kept to keep predicting meanwhile.
                        disp('Target lost. Switching back to detection...')
                        isTracking     = false;
                        missCount      = 0;
                        predFrameCount = 0;
                        confirmCount   = 0;
                        lastCenter     = [];
                        adaptiveQ      = 200;
                    end
                end
            end

        else
            % ===================== DETECTION MODE =====================
            if ~isempty(bboxesTmp)
                xcDet = bboxesTmp(1,1) + bboxesTmp(1,3)/2;
                ycDet = bboxesTmp(1,2) + bboxesTmp(1,4)/2;

                % Reject detections too far from the last known position
                isValid = true;
                if ~isempty(lastBboxes)
                    xcLast = lastBboxes(1) + lastBboxes(3)/2;
                    ycLast = lastBboxes(2) + lastBboxes(4)/2;
                    dist   = norm([xcDet - xcLast, ycDet - ycLast]);
                    if dist > imageSizeTracking * 2.0
                        disp('Warning: False positive rejected')
                        isValid      = false;
                        confirmCount = 0;
                        bboxes       = lastPredBbox;   % keep the frozen bbox
                    end
                end

                if isValid
                    % Valid detection: switch to confirmation phase
                    disp('Target detected. Waiting for confirmation...')
                    lastBboxes     = bboxesTmp(1,:);
                    bboxes         = bboxesTmp;
                    isTracking     = true;
                    isConfirming   = true;
                    confirmCount   = 1;
                    missCount      = 0;
                    predFrameCount = 0;
                    lastPredBbox   = [];
                end
            else
                % Nothing detected on the full frame
                confirmCount = 0;

                if kfInitialized
                    missCount = missCount + 1;
                    if missCount >= maxMisses * 3
                        % Target lost for too long: fully reset the tracker
                        disp('No target found. Resetting tracker...')
                        kfInitialized  = false;
                        lastBboxes     = [];
                        lastPredBbox   = [];
                        bboxes         = [];
                        missCount      = 0;
                        predFrameCount = 0;
                        lastCenter     = [];
                        adaptiveQ      = 200;
                    end
                end
            end
        end
    end

    % 5 - Draw the bbox on the full frame
    % Colors: yellow = confirming, green = frozen (target lost), red = tracking
    if ~isempty(bboxes)
        if isConfirming
            frameDisplay = insertShape(frame, 'Rectangle', bboxes, 'Color', 'yellow');
        elseif ~isTracking && predFrameCount >= maxPredFrames
            frameDisplay = insertShape(frame, 'Rectangle', bboxes, 'Color', 'green');
        else
            frameDisplay = insertShape(frame, 'Rectangle', bboxes, 'Color', 'red');
        end
    else
        frameDisplay = frame;
    end

    step(player, frameDisplay)

    % Keep playback at the video's real frame rate
    frameTime = toc;
    if frameTime < dt
        pause(dt - frameTime)
    end
end

%% Cleanup
release(player)
if ~isempty(futureInference)
    cancel(futureInference);   % stop any pending inference
end
fprintf('\n')