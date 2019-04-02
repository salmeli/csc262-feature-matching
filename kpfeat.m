function [descriptors] = kpfeat(img,keypoints,patch_size,downsampling_factor)

    % add default parameters
    if(nargin < 4)
        downsampling_factor = 5;
    end
    if(nargin < 3)
        patch_size = 8;
    end

    % find indices of keypoint
    [fcol, frow] = find(keypoints >= 1);
    
    % allocate space for result
    descriptors(size(fcol, 1), patch_size ^ 2) = 0;
    
    % blur and downsample input image
    gauss_5 = gkern(5^2);
    blurred_img = conv2(gauss_5, gauss_5, img, 'same');
    blurred_img = blurred_img(1:downsampling_factor:end, ...
        1:downsampling_factor:end);

    % loop through all features and extract a patch
    for k=1:size(fcol,1)
        % get coordinates of feature in downsampled image
        kRow = uint8(frow(k)/downsampling_factor);
        kCol = uint8(fcol(k)/downsampling_factor);
        
        % if even patch size
        if (rem(patch_size,2) == 0)
            ul_offset = patch_size/2;
            lr_offset = ul_offset - 1;
        else 
            ul_offset = floor(patch_size / 2);
            lr_offset = ul_offset;
        end
        
        % calculate coordinates of corners of patch
        ul_corner_x = (kRow - ul_offset);
        ul_corner_y = (kCol - ul_offset);
        lr_corner_x = (kRow + lr_offset);
        lr_corner_y = (kCol + lr_offset);
        
        [img_width, img_height] = size(blurred_img);
        
        % check that patch does not fall outside image bounds
        if(ul_corner_x < 1 || ul_corner_y < 1 || ...
                lr_corner_x > img_width || lr_corner_y > img_height)
            descriptors(k,:) = NaN;
            continue;
        end
        
        % extract the patch
        patch = blurred_img(ul_corner_x:lr_corner_x, ...
            ul_corner_y:lr_corner_y);

        % normalize bias and gain of patch
        bias_norm_patch = patch - mean(patch(:));
        norm_patch = bias_norm_patch ./ std(bias_norm_patch(:));
        
        % store into array
        descriptors(k,:) = norm_patch(:);
    end  
end

