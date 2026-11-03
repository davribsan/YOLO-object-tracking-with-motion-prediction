function [anchorGrid, stride] = make_anchors(feats, strideValues)
% Creates a set of anchor coordinates and stride values for the given 
% feature maps. Used in object detection networks to map feature grid 
% positions back to the input image space.

% feats - cell array containing output features 
% gridCellOffset = 0.5 - value to shift the z,y values by 

    gridCellOffset = 0.5; 
    n = 3;
    anchorGridTmp = cell(n,1);
    strideTmp = cell(n,1);
    totalSize = 0;
    
    for i = 1:n
        sz = size(feats{i});
        totalSize = totalSize + (sz(1).*sz(2));
        anchorGridTmp{i,1} = coder.nullcopy(zeros(sz(1).*sz(2),2));
        strideTmp{i,1} = coder.nullcopy(zeros(sz(1).*sz(2),2));
    end
    
    for i=1:size(strideValues,2)
        [h,w,~,~]= size(feats{i});
        sx = (0:w-1)+gridCellOffset;
        sy = (0:h-1)+gridCellOffset;
        [sy,sx]= meshgrid(sy,sx);
        anchorGridTmp{i,1} = cat(2, sx(:), sy(:));
        strideTmp{i,1} = repmat(strideValues(i),h*w,1);
    end
    anchorGrid = cat(1,anchorGridTmp{:});
    stride = cat(1,strideTmp{:});

    
end