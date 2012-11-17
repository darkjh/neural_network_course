function visual( imageVector, imageNum )
    figure(imageNum);
    imageMatrix = reshape(imageVector,256,256);
    imagesc(imageMatrix);
    colormap gray;
end

