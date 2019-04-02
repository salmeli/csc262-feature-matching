% Output: A cell containing the following values in order: -
%   [ 1. ]  Points with Features
%   [ 2. ]  Determinant
%   [ 3. ]  Trace
%   [ 4. ]  Determinant/Trace Measure
%   [5,6.]  Partial Derivates w.r.t x and y
%   [7,8.]  Blurred Versions of Square of 5,6
%   [ 9. ]  Blurred Product of 5,6
%   [10. ]  Orientation
function results = kpdet(img, threshold, orientation_std_dev)

% If no threshold argument is given, set it to a default value
if (nargin < 3) 
    orientation_std_dev = 4.5;
end
if (nargin < 2) 
    threshold = 0.001;
end

% blurring_variance = 1.5;
blurring_std_dev = 1.5;

% Create the Gaussian (sigma = 1) and it's derivative
gauss_1 = gkern(1);
dgauss_1 = gkern(1, 1);

% Create the blurring Gaussian (sigma = 1.5)
gauss_1_5 = gkern(blurring_std_dev^2);

% Convolve to get the partial x and y gradient of the image
I_x = conv2(gauss_1, dgauss_1, img, 'same');
I_y = conv2(dgauss_1, gauss_1, img, 'same');

% Square of the partial derivatives
I_x2 = I_x .^ 2;
I_y2 = I_y .^ 2;
% Product of the partial derivatives
I_xy = I_x .* I_y;

% Blur the squared partial derivatives squared and also their product
I_x2_b = conv2(gauss_1_5, gauss_1_5, I_x2, 'same');
I_y2_b = conv2(gauss_1_5, gauss_1_5, I_y2, 'same');
I_xy_b = conv2(gauss_1_5, gauss_1_5, I_xy, 'same');

% The determinant of a 2x2 matrix is ac - bd
det_A = (I_x2_b .* I_y2_b) - (I_xy_b .^ 2);

% The trace of a 2x2 matrix is the sum of the diagonals
trace_A = (I_x2_b + I_y2_b);

% The measure of uniqueness for a point given by Richard Szeliski
det_trace_A = det_A ./ trace_A;

% Binarize the measure of uniqueness using a certain threshold
bin_det_trace_A = (det_trace_A > threshold);

% the maximal points of the binarized image
maxes_dt = maxima(bin_det_trace_A);

% Calculating the orientation of the gradient { Extra Credit Part 1 }
gauss_o = gkern(orientation_std_dev^2);
dgauss_o = gkern(orientation_std_dev^2, 1);
O_x = conv2(gauss_o, dgauss_o, img, 'same');
O_y = conv2(dgauss_o, gauss_o, img, 'same');
orientation = atan2(O_y, O_x);

% Return certain features
results = { maxes_dt, det_A, trace_A, det_trace_A, ...
            I_x, I_y, I_x2_b, I_y2_b, I_xy_b, orientation };
end

