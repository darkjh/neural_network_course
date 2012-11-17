%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here are some useful line codes for 1st mini project
% Written by Mohammadjavad FARAJI and Felipe GERHARD, 
% 14.10.2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % How to read grayscale images
% img = imread('i1.gif');

% % How to know size of the image
% dim = size(img,1);

% % How to change the format from uint8 to double
% img = double(img);

% % How to change the range from [0 255] to [0 1]
% img = double(img)./255;

% % How to convert an image to a row vector
% img_vec = img(:)';

% % How to reshape from vector to matrix
% img = reshape(img_vec, dim, dim);

% % How to show an image scaled to [0 1]
% imshow(uint8(255*img))

% % How to mix n row images scaled to [0 1]
% mix_imgs = rand(n,n) * [img_1; ... ; img_n]

% % How to remove the mean from a row vector 
% vec = vec - repmat(mean(vec),1,length(vec));

% % How to white Data = [vec_1; ... ; vec_n]
% White_Data = whitendata(Data);

