function newCoords = postProcessYOLOv8_fixed(imgShape,coords)
% Ensure that all coordinates are valid within the original image boundaries.

    newCoords = coords;
    newCoords(newCoords < 0) = 0.1;
    newCoords(:,1) = min(newCoords(:,1), imgShape(2));
    newCoords(:,2) = min(newCoords(:,2), imgShape(1));
    newCoords(:,3) = min(newCoords(:,3), imgShape(2));
    newCoords(:,4) = min(newCoords(:,4), imgShape(1));

end