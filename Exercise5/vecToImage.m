function [] = vecToImage(img_vec)
% function does not return any value
% receives an "image" as a vector, calculates it's square root ==> sz.
%  uses reshape and imagesc to plot the image.

sz = sqrt(length(img_vec));
imagesc(reshape(img_vec, sz, sz));
end

