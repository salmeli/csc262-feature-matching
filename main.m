%% [CSC 262] Lab: Feature Matching
clearvars;
close all;

% THINGS TO FINISH
% (10 points) Sorted feature distances (B.7)
% (10 points) Feature distance observations (B.8)
% (15 points) Matched feature offset (B.9) and images (C.3)
% (10 points) Estimated alignment (E.2)
% (10 points) Aligned images (E.4)
% (20 points) Alignment observations (E.5)

%% Overview
%

% Choose between local or remote file
%img_1 = '/home/weinman/courses/CSC262/images/kpmatch1.png';
img_1 = 'kpmatch1.png';
%img_2 = '/home/weinman/courses/CSC262/images/kpmatch1.png';
img_2 = 'kpmatch2.png';

% Read in the images
img_kp1 = im2double(imread(img_1));
img_kp2 = im2double(imread(img_2));

%% Matching Set-up
%
 
% configuration
detection_threshold = 0.0001;

% locations of keypoints
kp1_locs_intermediate = kpdet(img_kp1, detection_threshold);
kp1_locs = kp1_locs_intermediate{1};
kp2_locs_intermediate = kpdet(img_kp2, detection_threshold);
kp2_locs = kp2_locs_intermediate{1};

clear kp1_locs_intermediate;
clear kp2_locs_intermediate;
 
% patches of keypoints that have been flattened
kp1_desc = kpfeat(img_kp1, kp1_locs);
kp2_desc = kpfeat(img_kp2, kp2_locs);
 
% indices of the keypoints
[f1_x, f1_y] = find(kp1_locs);
[f2_x, f2_y] = find(kp2_locs);

% allocate memory
num_features = size(kp1_desc, 1);
translations(num_features,2) = 0;
 
%% Feature Matching
% 
 
for k=1:num_features
    % flattened version of our extracted patch
    chosen_feature = kp1_desc(k,:);

    % if feature is invalid, skip over it
    if (isnan(chosen_feature))
        translations(k,:) = NaN;
        continue;
    end    

    % distances between the chosen feature in the first image
    % and all the features in the other image
    distance = kp2_desc - chosen_feature;
    euc_distances(size(distance,1)) = 0;

    % calculating the Euclidean distance from the current distances
    % for each feature point in the second image that is being matched
    % with the chosen feature in the first image
    for i=1:size(distance, 1)
        cur_distance = distance(i);

        % skip over features that do not exist
        if (isnan(cur_distance))
            euc_distances(i) = NaN;
            continue;
        end

        cur_euc_distance = cur_distance .^ 2;
        cur_euc_distance = sum(cur_euc_distance(:));
        cur_euc_distance = sqrt(cur_euc_distance);

        euc_distances(i) = cur_euc_distance;
    end

    % sort the euclidean distances but keep the indices
    % (B.7)
    [euc_dist, euc_index] = sort(euc_distances);
    % graphing the quality-values of the closest 10 matches

    %figure;
    % display the value of the closest matches
    if (size(euc_distances, 2) > 10)
    %    bar(euc_dist(1:10))
    else
    %    bar(euc_dist)
    end

    %% (B.8)

    % index of the best matching feature
    bmf_index = euc_index(1);
    bmf_x = f2_x(bmf_index);
    bmf_y = f2_y(bmf_index);

    % Show the best match
    %figure;
    %imshowpair(img_kp1, img_kp2, 'montage');
    %hold on;
    %plot(f1_y(k), f1_x(k), 'r*');
    %hold on;
    %plot(bmf_x + 352, bmf_y, 'r*');
    %hold on;
    %line([f1_y(k) (bmf_x + 352)],[bmf_y f1_x(k)]);

    % estimated translation
    trans_x = f1_x(k) - bmf_x;
    trans_y = f1_y(k) - bmf_y;
    
    translations(k,:) = [trans_x trans_y];
end