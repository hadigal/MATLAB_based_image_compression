%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPE_565_HW1
% Author: Hrishikesh Adigal, Kishore Bharathkumar
% email : <Hrishikesh:hadigal@sdsu.edu>, <Kishore:kishorebharath14@gmail.com>
% Date: 02/17/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;
%QUESTION-1: READ AND DISPLAY THE IMAGE

hkbimg=imread('~/Flooded_house.jpg','jpg');
figure(1);
imshow(hkbimg);
title('Real Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%QUESTION-2: DISPLAY R G B BANDS OF THE ORIGINAL IMAGE

%EXTRACTING R G B COMPONENTS
red=hkbimg;
red(:,:,2:3)=0;
green=hkbimg;
green(:,:,[1 3])=0;
blue=hkbimg;
blue(:,:,1:2)=0;
%DISPLAYING R G B COMPONENTS
figure(2);
subplot(2,2,1);imshow(hkbimg);title('Real image');
subplot(2,2,2);imshow(red);title('Red');
subplot(2,2,3);imshow(green);title('Green');
subplot(2,2,4);imshow(blue);title('Blue');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%QUESTION-3.1: RGB TO YCbCr IMAGE DISPLAY
yccimg=rgb2ycbcr(hkbimg);
figure(3);
imshow(yccimg);
title('RGB 2 YCbCr');

%QUESTION-3.2: YCbCr to RGB IMAGE DISPLAY
h=ycbcr2rgb(yccimg);
figure(4);
imshow(h);
title('YCbCr 2 RGB');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%QUESTION-4: EXTRACT AND DISPLAY Y Cb Cr COMPONENTS

y=yccimg(:,:,1);
cb=yccimg(:,:,2);
cr=yccimg(:,:,3);

figure(5);
subplot(2,2,1);imshow(yccimg);title('Real YCbCr image');
subplot(2,2,2);imshow(y);title('Y');
subplot(2,2,3);imshow(cb);title('Cb');
subplot(2,2,4);imshow(cr);title('Cr');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%QUESTION-5: SUBSAMPLE AND DISPLAY Cb Cr BANDS USING 4:2:0

ycc_size = size(yccimg);
fprintf("Size of original ycbcr image:%s\n",mat2str(ycc_size));
cb_comp = cb;
cr_comp = cr;

total_rows = size(cb,1);
total_col  = size(cr,2);

% Making the even rows and cols zero
for row = 2:2:total_rows
    for col = 2:2:total_col
        cb_comp(row,col) = 0;
        cr_comp(row,col) = 0;
    end
end

% Now subsample Chrominance by 4:2:0 of the pixels by taking
% out the alternate pixels from both rows and columns
cb_420 = cb_comp(1:2:end,1:2:end);
cr_420 = cr_comp(1:2:end,1:2:end);

% Displaying the Sub Sampled Cb and Cr components and comparing with
% original Cb and Cr components
figure(6)
subplot(2,2,1); subimage(cb); title('Original Cb Component');
subplot(2,2,2); subimage(cr); title('Original Cr Component');
subplot(2,2,3); subimage(cb_420); title('4:2:0 Sub Sampled Cb Component');
subplot(2,2,4); subimage(cr_420); title('4:2:0 Sub Sampled Cr Component');

cb_comp_size = size(cb);
cr_comp_size = size(cr);
cb_comp_420_size = size(cb_420);
cr_comp_420_size = size(cr_420);

fprintf("Size of Original Cb Component:%s\n",mat2str(cb_comp_size));
fprintf("Size of 4:2:0 Sub Sampled Cb Component:%s\n",mat2str(cb_comp_420_size));
fprintf("Size of Original Cr Component:%s\n",mat2str(cr_comp_size));
fprintf("Size of 4:2:0 Sub Sampled Cr Component:%s\n",mat2str(cr_comp_420_size));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-6: UPSAMPLE AND DISPLAY Cb Cr COMPONENTS USING

% 6.1: LINEAR INTERPOLATION
% Make copy of the Cb and Cr components of the image with
% even row and columns pixels values set to zero.
lin_int_cb = cb_comp;
lin_int_cr = cr_comp;

% Checking if there are even numbered rows and columns
if mod(total_rows,2) == 0
    % Iterating over the even # of rows and columns:
    for cbcr_row = 2:2:total_rows - 2
        for cbcr_col = 2:2:total_col - 2
            lin_int_cb(cbcr_row,cbcr_col) = round(lin_int_cb(cbcr_row -1,cbcr_col - 1)/2 + lin_int_cb(cbcr_row + 1,cbcr_col + 1)/2);
            lin_int_cr(cbcr_row,cbcr_col) = round(lin_int_cr(cbcr_row -1,cbcr_col - 1)/2 + lin_int_cr(cbcr_row + 1,cbcr_col + 1)/2);
        end
    end
    % Since the last pixel's column and row (r,c) doesn't have a (r+1,c+1)
    % row and a column component for performing Linear Interpolation, by
    % using Symmetrical Extension, we copy the previous row and column for (r,c)
    for cbcr_row = total_rows
        for cbcr_col = total_col
            lin_int_cb(cbcr_row,cbcr_col) = lin_int_cb(cbcr_row-1,cbcr_col-1);
            lin_int_cr(cbcr_row,cbcr_col) = lin_int_cr(cbcr_row-1,cbcr_col-1);
        end
    end
% This fragment will be executed if the rows and columns are odd in number.
else
    for cbcr_row = 2:2:total_rows -1
        for cbcr_col = 2:2:total_col -1
           lin_int_cb(cbcr_row,cbcr_col) = round((lin_int_cb(cbcr_row -1,cbcr_col - 1) + lin_int_cb(cbcr_row + 1,cbcr_col + 1))/2);
           lin_int_cr(cbcr_row,cbcr_col) = round((lin_int_cr(cbcr_row -1,cbcr_col - 1) + lin_int_cr(cbcr_row + 1,cbcr_col + 1))/2);
        end
    end
end

figure(7)
subplot(2,2,1), imshow(cb),title('Original Cb component')
subplot(2,2,2), imshow(lin_int_cb),title('Cb Component Upsampled by Linear Interpolation')
subplot(2,2,3), imshow(cr),title('Original Cr component')
subplot(2,2,4), imshow(lin_int_cr),title('Cr Component Upsampled by Linear Interpolation')

% Concatenating the Y,Cb and Cr components to form the Y Cb Cr
% Up Sampled Image by Linear Interpolation
upsampled_ycbcr_lin_int = cat(3,y,lin_int_cb,lin_int_cr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 6.2: SIMPLE ROW OR COLUMN REPLICATION
% Copying (n-1)th row to the Even numbered row.
rc_replica_cb = cb;
rc_replica_cb(2:2:end,2:2:end) = rc_replica_cb(1:2:end,1:2:end);

% Copying (n-1)th column to the Even numbered column.
rc_replica_cr = cr;
rc_replica_cr(2:2:end,2:2:end)=rc_replica_cr(1:2:end,1:2:end);

figure(8)
subplot(2,2,1), imshow(cb),title('Original Cb component')
subplot(2,2,2), imshow(rc_replica_cb),title('Cb Component Upsampled by RC Replication')
subplot(2,2,3), imshow(cr),title('Original Cr component')
subplot(2,2,4), imshow(rc_replica_cr),title('Cb Component Upsampled by RC Replication')

% Concatenating the Y,Cb and Cr components to form the Y Cb Cr
% Up Sampled Image by Simple Row Column Replication
upsampled_ycbcr_rc_replica = cat(3,y,rc_replica_cb,rc_replica_cr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-7: CONVERT THE UPSAMPLED Y Cb Cr IMAGES TO RGB IMAGE

upsampled_rgb_lin_int = ycbcr2rgb(upsampled_ycbcr_lin_int);
upsampled_rgb_rc_replica = ycbcr2rgb(upsampled_ycbcr_rc_replica);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-8: DISPLAY THE ORIGINAL AND RECONSTRUCTED IMAGES (The image restored from the YCbCr coordinates)

figure(9)
subplot(2,2,1), imshow(hkbimg), title('Original RGB Image')
subplot(2,2,2), imshow(upsampled_rgb_lin_int), title('Upsampled RGB Image using Linear Interpolation')
subplot(2,2,3), imshow(upsampled_rgb_rc_replica), title('Upsampled RGB Image using Simple RC Replication')

size_upsampled_ycbcr = size(upsampled_ycbcr_lin_int);
size_upsampled_rgb = size(upsampled_rgb_lin_int);

fprintf("Dimensions of Upsampled YCbCr image using Linear Interpolation: %s\n",mat2str(size_upsampled_ycbcr));
fprintf("Dimensions of Upsampled RGB image using Linear Interpolation: %s\n",mat2str(size_upsampled_rgb));

size_upsampled_ycbcr = size(upsampled_ycbcr_rc_replica);
size_upsampled_rgb = size(upsampled_rgb_rc_replica);

fprintf("Dimensions of Upsampled YCbCr image using Simple Row Column Replication: %s\n",mat2str(size_upsampled_ycbcr));
fprintf("Dimensions of Upsampled RGB image using Simple Row Column Replication: %s\n",mat2str(size_upsampled_rgb));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-9: COMMENT ON THE VISUAL QUALITY OF THE RECONSTRUCTED IMAGES FOR BOTH THE UPSAMPLING CASES

% From the reconstructed it can be observed that the image upsampled using
% LINEAR INTERPOLATION has a better quality as compared to the image
% upsampled using ROW COLUMN REPLICATION. This is because in the RC
% Replication, the pixel of 'n-1' column or row is extended to 'n' row or
% column and so the pixels are large and jagged. In Linear Interpolation,
% the average of the neighbouring column or row is substituted at the
% missing pixels row or column, thus making it look more clear.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-10: MEASUREMENT OF MEAN SQUARED ERROR BETWEEN THE ORIGINAL
% AND RECONSTRUCTED RGB IMAGE OBTAINED BY LINEAR INTERPOLATION

% MSE is calucated using the formula:
% (1/N*M)*sum_of_{[(N,M)_org_image - (N,M)_reconstructed_image)]^2}

diff = (hkbimg - upsampled_rgb_lin_int);
mse = (sum(sum(diff.^2)))/(total_rows*total_col);

fprintf('R component MSE: %f\n', mse(:,:,1));
fprintf('G component MSE: %f\n', mse(:,:,2));
fprintf('B component MSE: %f\n', mse(:,:,3));

% COMMENT:
% R component MSE: 3.385195
% G component MSE: 1.595923
% B component MSE: 5.933782
% There seems to be slight distortion in Green component of the
% reconstructed image and the maximum distortion is seen in the
% Blue component of the image.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% QUESTION-11: COMPRESSION RATIO OBTAINED BY 4:2:0 SUBSAMPLING OF Cb Cr COMPONENTS

% Calculating the total size in bits for Original image.
org_img_size = size(yccimg,1)*size(yccimg,2)*3;
% Calculating the total size in bits for Subsampled image.
sampled_img_420 = size(y,1)*size(y,2) + size(cb_420,1)*size(cb_420,2) + size(cr_420,1)*size(cr_420,2);

% Compression Ratio (CR) is calculated using the formula
% (Total File Size of Original Image)/(Total File Size of Subsampled Image)
CR = org_img_size/sampled_img_420;

% Printing the CR value on the Command Window
fprintf('\nFile size of Original YCbCr image: %d',(org_img_size)/8);
fprintf('\nFile size of 4:2:0 YCbCr Subsampled image: %d',(sampled_img_420)/8);
fprintf('\nCompression Ratio(CR): %f\n',CR);

% The CR value obtained is 2:1 which implies that the Original image can be
% compressed from a size of (141,054) Bytes File to (70752) Bytes File.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
