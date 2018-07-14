%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: README_HW4.m
% Date: 04/28/2018
% Author: Hrishikesh Adigal, Kishore Bharatkumar, Jameson Thies, Kunal Mehta
% email : <Hrishikesh:hadigal@sdsu.edu>, <Kishore:kishorebharath14@gmail.com>
% <Jameson:jamesonthies@gmail.com>, <Kunal kmehta@sdsu.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

% the quantization matrix for luminance from lecture
lum_q_matrix = [16 11 10 16 24 40 51 61;
                12 12 14 19 26 58 60 55;
                14 13 16 24 40 57 69 56;
                14 17 22 29 51 87 89 62;
                18 22 37 56 68 109 103 77;
                24 35 55 64 81 104 113 92;
                49 64 78 87 108 121 120 101;
                72 92 95 98 112 100 103 99];

% the quantization marix for chromiance from lecture
chr_q_matrix = [17 18 24 47 99 99 99 99;
                18 21 26 66 99 99 99 99;
                24 26 56 99 99 99 99 99;
                47 66 99 99 99 99 99 99;
                99 99 99 99 99 99 99 99;
                99 99 99 99 99 99 99 99;
                99 99 99 99 99 99 99 99;
                99 99 99 99 99 99 99 99];

% Reading the football video sequence.
v_obj = VideoReader('football_qcif.avi');
i = 0;
j = 1;
width = v_obj.Width;
height = v_obj.Height;
fprintf("Dimensions of the Video sequence: [height*width]:[%d x %d]\n",height,width);

% extracting every frame from the given video sequence and coverting each
% extracted frame to ycbcr.
while hasFrame(v_obj)
    v_frm(j).cdata = readFrame(v_obj);
    ycbcr = rgb2ycbcr(v_frm(j).cdata(:,:,:));
    y(:,:,j) = ycbcr(:,:,1);
    cb(:,:,j) = ycbcr(1:2:end,1:2:end,2);
    cr(:,:,j) = ycbcr(1:2:end,1:2:end,3);
    %cal the # of frames in the video sequence
    i = i+1;
    j = j+1;
end
fprintf("Total Frames in the video sequence:%d\n",i);

% Defining the Frame 2 i.e the first frame from given GoP(2:6)[IPPPP] as
% intra-frame
rf = y(:,:,2);
figure(1)
imshow(rf)
title('I-Frame')

% computing total MB possible for the given video sequence.
t_mb = (ceil(size(rf,1)/16))*(ceil(size(rf,2)/16));
fprintf("Total MB possible:%d\n",t_mb);

% Var to compute the total comparisons done using the below algorithm.
comp = 0;
t_blk = (ceil(size(rf,1)/8))*(ceil(size(rf,2)/8));
[y_r,y_c] = size(rf);

%array to hold temp reconst. frame
temp_i_frm = zeros(y_r,y_c);
[ch_r,ch_c] = size(cb(:,:,1));
temp_cb = zeros(ch_r,ch_c);

% DCT and quant on I frame
[y_dct,cb_dct,cr_dct,y_q_dct,cb_q_dct,cr_q_dct] = dct_qaunt(rf,cb(:,:,2),cr(:,:,2),lum_q_matrix,chr_q_matrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate AC and DC coefficients
%arrays to ac and dc coefficients for 5 frames
ac_y = zeros(5,(y_r/8)*(y_c/8)*63);
dc_y = zeros(5,(y_r/8)*(y_c/8));
ac_r = 1;

% using the function dc_ac_per_blk() defined below to compute DC and AC
% coefficients
[dc_y,ac_y,dc_c,ac_c] = dc_ac_per_blk(y_q_dct,dc_y,ac_y,ac_r);
fprintf("\nTotal DC coefficients for I frame stored in R:%d of dc_y array:%d\nTotal AC coefficients for I frame stored in R:%d of ac_y array:%d\n",ac_r,dc_c-1,ac_r,ac_c-1);
fprintf("\nThe DC coefficients for 1st 8x8 block of the I-frame:%d\n",dc_y(ac_r,1));
fprintf("\nThe AC coefficients for 1st 8x8 block of the I-frame using zig-zag tranform:\n");
display(ac_y(ac_r,1:63))

ac_r = ac_r + 1;
% i-DCT and i-qaunt on I frame
[iq_y,iq_cb,iq_cr,y_idct,cb_idct,cr_idct] = inv_dct_quant(y_q_dct,cb_q_dct,cr_q_dct,lum_q_matrix,chr_q_matrix);

% Sending the I frame to the decoder and reconstructing the I frame at the
% decoder
fc = 1;
d_f(:,:,:,fc) = decoder(y_q_dct,cb_q_dct,cr_q_dct,lum_q_matrix,chr_q_matrix);
fc = fc+1;

% storing the i-dct y frame at the encode to a temp array.
temp_i_frm = y_idct;
% array matrix to store predicted frame
ref_frm = temp_i_frm;
% array to update the predicted frame as the MB window keeps increasing.
diff_frm = zeros(y_r,y_c);

% iterating through the loop to find difference matrix,motion estimation
% and motion vector
for i = 3:6
    % Extracting the Y,cb and Cr compoments of downsampled image frames of
    % the video frame image.
    % setting the current and reference frames for motion prediction
    tgt_frm = y(:,:,i);
    tgt_cb = cb(:,:,i);
    tgt_cr = cr(:,:,i);

    % temp var to hold the target and ref. frames
    tgt_frm_2 = double(tgt_frm);
    ref_frm_2 = double(ref_frm);
    % defining the MB size as 16 for Y comp.
    mb_size = 16;
    % getting total rows and column in the current frame
    [row,col] = size(tgt_frm_2);
    % used for storing the diff frame MB
    temp_diff_sw = zeros(mb_size,mb_size);
    % array to hold displacement vector
    disp_vect = zeros(1,2);
    % two matrices for holding the diff frame and tgt frame
    search_window = zeros(t_mb,2,2);
    % var to compute the search window movement.
    c = 1;
    % iterating over the intire image frame at MB intervals
    for r_mb = 1:mb_size:row
        for c_mb = 1:mb_size:col
            % Extracting the temporary target frame window.
            temp_tf_mb = tgt_frm_2(r_mb:r_mb+15,c_mb:c_mb+15);
            max_val = 66000;
            for blk_r = -8:8
                for blk_c = -8:8
                    % Now padding the extra rows and columns inside the
                    % search window
                    inc_row = r_mb+blk_r;
                    inc_col = c_mb+blk_c;
                    if ((inc_row + 15) <= row) && ((inc_col +15) <= col) && (inc_row > 0)  && (inc_col > 0)
                        rf_mb_sw = ref_frm_2(inc_row:inc_row+15,inc_col:inc_col+15);
                        temp_diff_sw = temp_tf_mb - rf_mb_sw;
                        % computing the MSE for current MB and reference MB
                        % within the computed search window
                        mse_mb = sum(sum(temp_diff_sw.^2));
                        mse_mb = mse_mb./256;
                        % Finding the closest block matching also computing
                        % the MAD
                        if mse_mb < max_val
                            max_val = mse_mb;
                            % calculating the motion vector for current
                            % frame with respect to previous frame.
                            disp_vect = [inc_row - r_mb,inc_col - c_mb];
                            % predic the value for current frame with
                            % respect to previous frame using MB SW
                            rc_img(r_mb:r_mb+mb_size -1,c_mb:c_mb+mb_size -1) = ref_frm_2(inc_row:inc_row+mb_size -1,inc_col:inc_col+mb_size -1);
                            diff_frm(r_mb:r_mb+mb_size -1,c_mb:c_mb+mb_size -1) = temp_diff_sw;
                            % incrementing the comparison count to display
                            % total MAD comaprisons at the end of the loop
                            comp = comp +1;
                        elseif (mse_mb == max_val)
                            pad_r_c = (r_mb - inc_row)^2 + (c_mb - inc_col)^2;
                            if pad_r_c < max_val
                                disp_vect = [inc_row - r_mb,inc_col - c_mb];
                            end
                        end
                    end
                end
            end
            % Now updating the search window positions
            search_window(c,:,1) = [r_mb,c_mb];
            search_window(c,:,2) = disp_vect;
            c = c+1;
        end
    end

    figure()
    % displaying the Motion Vector Representation using the quiver command
    quiver(search_window(:,2,1), search_window(:,1,1), search_window(:,2,2), search_window(:,1,2));
    title(['Motion vector for Y Component. Frame: ',num2str(i)]);
    grid on
    % performing the DCT and I-DCT on the difference frame
    % DCT and quant on Difference frame
    dct_diff_frm = uint8(diff_frm);
    [dy_dct,dcb_dct,dcr_dct,dy_q_dct,dcb_q_dct,dcr_q_dct] = dct_qaunt(dct_diff_frm,0,0,lum_q_matrix,chr_q_matrix);
    % i-DCT and i-qaunt on Difference frame
    [iq_dy,iq_dcb,iq_dcr,dy_idct,dcb_idct,dcr_idct] = inv_dct_quant(dy_q_dct,0,0,lum_q_matrix,chr_q_matrix);

    %Displaying the predicted, original and predicted error frames
    reconst_img = uint8(rc_img);
    figure;
    subplot(3,1,1),imshow(tgt_frm),title(['Orginal Y Component. Frame: ',num2str(i)])
    subplot(3,1,2),imshow(reconst_img),title(['Predicted Y Component. Frame: ',num2str(i)])
    subplot(3,1,3),imshow(dct_diff_frm),title('Difference')
    pred_err = tgt_frm - reconst_img;
    %subplot(3,1,3),imshow(pred_err),title('Difference')
    mse_img = sum(sum(pred_err.^2));
    mse_img = mse_img./(row*col);
    fprintf("\nThe MSE between original and predicted Y-comp. of Frame#%d: %f\n",i,mse_img);

    %updating the motion predicted image storage with current predicted frame
    ref_frm = reconst_img;

    % performing the DCT and I-DCT on the predicted frame frame
    % DCT and quant on the predicted frame frame
    [ry_dct,rcb_dct,rcr_dct,ry_q_dct,rcb_q_dct,rcr_q_dct] = dct_qaunt(ref_frm,tgt_cb,tgt_cr,lum_q_matrix,chr_q_matrix);

    % Now finding the DC and AC coefficients of perdicted frame and
    % updating the dc and ac coefficient array.
    [dc_y,ac_y,dc_c,ac_c] = dc_ac_per_blk(ry_q_dct,dc_y,ac_y,ac_r);
    fprintf("\nTotal DC coefficients for p-frame#%d stored in R:%d of dc_y array:%d\nTotal AC coefficients for p-frame#%d stored in R:%d of ac_y array:%d\n",i,ac_r,dc_c-1,i,ac_r,ac_c-1);
    %fprintf("\nAC:\n")
    %display(ac_y(a_r,1:63))
    %fprintf("\nDC:\n")
    %display(dc_y(a_r,1))
    ac_r = ac_r + 1;

    % i-DCT and i-qaunt on the predicted frame frame
    [iq_ry,iq_rcb,iq_rcr,ry_idct,rcb_idct,rcr_idct] = inv_dct_quant(ry_q_dct,0,0,lum_q_matrix,chr_q_matrix);
    ref_frm = ry_idct;

    % Sending the I frame to the decoder and reconstructing the I frame at the
    % decoder
    d_f(:,:,:,fc) = decoder(ry_q_dct,rcb_q_dct,rcr_q_dct,lum_q_matrix,chr_q_matrix);
    fc = fc +1;
end

figure;
imshow(d_f(:,:,:,1)), title("reconstructed I1 frame 2")
figure;
imshow(d_f(:,:,:,2)), title("reconstructed P2 frame 3")
figure;
imshow(d_f(:,:,:,3)), title("reconstructed P3 frame 4")
figure;
imshow(d_f(:,:,:,4)), title("reconstructed P4 frame 5")
figure;
imshow(d_f(:,:,:,5)), title("reconstructed P5 frame 6")

[add,sub] = exhaustive_search_load_cal(16);
fprintf("\nTotal additions for while computing:%d\nTotal substractions for while computing:%d\nTotal comparisons for while computing:%d\n",add,sub,comp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: exhaustive_search_load_cal()
% imput: (int) macro block size
% Return: (int_array) returns total additions and substractions performed
% during computation of motion estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [t_add,t_sub] = exhaustive_search_load_cal(mb_size)
    t_s = ((2*8)+1)^2;
    t_add = (2*(mb_size^2))*t_s;
    t_sub = (mb_size^2)*t_s;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: zigzag(q_blk))
% Discription: the zigzag function to extract AC coefficients from the 8x8
% quantized block
% input: (int)8x8 block
% Return: (int_array) returns an array of AC coeffiecnts of the 8x8
% quantized block.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ac_mat = zigzag(q_blk)
index=1;
mat_dct = q_blk;
ac_mat = [];
N = 8;
for c=1:2*N-1
    if c<=N
        if mod(c,2)==0
            j=c;
            for i=1:c
                ac_mat(index)= mat_dct(i,j);
                index=index+1;j=j-1;
            end
        else
            i=c;
            for j=1:c
                ac_mat(index)= mat_dct(i,j);
                index=index+1;i=i-1;
            end
        end
    else
        if mod(c,2)==0
            p=mod(c,N); j=N;
            for i=p+1:N
                ac_mat(index)=mat_dct(i,j);
                index=index+1;j=j-1;
            end
        else
            p=mod(c,N);i=N;
            for j=p+1:N
                ac_mat(index)=mat_dct(i,j);
                index=index+1;i=i-1;
            end
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: dct_qaunt(y_comp,cb_420,cr_420)
% Discription: the dct_quant function 2D-DCT transform and quantiztion of the
% frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y_dct,cb_dct,cr_dct,y_q_dct,cb_q_dct,cr_q_dct] = dct_qaunt(y_comp,cb_420,cr_420,lum_q_matrix,chr_q_matrix)
%taking out 1st 8x8 block from the luminance comp.
%making a copy of y comp
y_cpy = y_comp;
% block processing with 2D DCT transform
% since dct2 function accepts matrix only as input need to make the
% blockproc function return appropriate size so double data type.

%dct_handle = @(block_struct)dct2(block_struct.data);
dct_handle = @dct2;
% now 8x8 block processing on y_cpy
y_dct = blkproc(y_cpy,[8 8],dct_handle);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now quantizing the luminance and chromiance matrix with the quantizer
% matrix from the lecture.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% creating copies of y,cb and cr dct matricies
y_dct_cpy  = y_dct;
% dividing the dct block structure with the lum matrix.
q_lum = @(y_dct_cpy)round(y_dct_cpy./lum_q_matrix);
% 8x8 block processing on the quantized matrix
y_q_dct = blkproc(y_dct_cpy,[8 8],q_lum);
% DCT and quantization for chromiance components
if (cb_420 ~= 0) | (cr_420 ~= 0)
    % now computing the DC the coefficients for the cb and cr components
    cb_cpy = cb_420;
    cr_cpy = cr_420;
    cb_dct = blkproc(cb_cpy,[8 8],dct_handle);
    cr_dct = blkproc(cr_cpy,[8 8],dct_handle);
    cb_dct_cpy = cb_dct;
    cr_dct_cpy = cr_dct;
    % dividing the dct block_struct data with the chromiance matrix.
    q_chr1 = @(cb_dct_cpy)round(cb_dct_cpy./chr_q_matrix);
    q_chr2 = @(cr_dct_cpy)round(cr_dct_cpy./chr_q_matrix);
    % 8x8 block processing on the quantized matrix
    cb_q_dct = blkproc(cb_dct_cpy,[8 8],q_chr1);
    cr_q_dct = blkproc(cr_dct_cpy,[8 8],q_chr2);
else
    cb_dct   = 0;
    cr_dct   = 0;
    cb_q_dct = 0;
    cr_q_dct = 0;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: inv_dct_quant(y_q,cb_q,cr_q)
% Discription: the idct_quant function performs 2D-iDCT transform and
% i-quantiztion of the frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [iq_y,iq_cb,iq_cr,y_idct,cb_idct,cr_idct] = inv_dct_quant(y_q,cb_q,cr_q,lum_q_matrix,chr_q_matrix)
% in inverse quantization we will multiply the quantization lum and chr.
% matrix to the quantized dct matricies of the lum and chromiance
% components.

% now multiplying the quantized lum matirx with the quantization matrix for
% inverse dct operation

y_q_cpy  = y_q;
cb_q_cpy = cb_q;
cr_q_cpy = cr_q;

% performing the inverse quantization by multiplying the quantization
% matrix to get the inverse quantized matrices
iq_lum = @(y_q_cpy)round(y_q_cpy.*lum_q_matrix);
% 8x8 block processing of the entire matrix
iq_y = blkproc(y_q,[8 8],iq_lum);

% performing the same operation for cb and cr
iq_chr1 = @(cb_q_cpy)round(cb_q_cpy.*chr_q_matrix);
iq_chr2 = @(cr_q_cpy)round(cr_q_cpy.*chr_q_matrix);

iq_cb = blkproc(cb_q,[8 8],iq_chr1);
iq_cr = blkproc(cr_q,[8 8],iq_chr2);
% now will apply the inverse DCT to y cb and cr component matrices
% separately.

idct_handle = @(block_struct)idct2(block_struct.data);
% using the above handle with the blockproc()
y_idct  = blockproc(iq_y,[8 8],idct_handle);
cb_idct = blockproc(iq_cb,[8 8],idct_handle);
cr_idct = blockproc(iq_cr,[8 8],idct_handle);

%since ycbcr needs to be in the unsigned int format typecasting the above
%array to uint16

y_idct  = uint8(y_idct);
cb_idct = uint8(cb_idct);
cr_idct = uint8(cr_idct);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: lin_int_rc(y,cb,cr,t_r,t_c)
% Discription: Perfrom linear interpolation for frame reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [rep_ycbcr] = lin_int_rc(y,cb,cr,t_r,t_c)

rep_ycbcr = zeros(t_r,t_c,3);

cb_idct_cpy = cb;
cr_idct_cpy = cr;

rep_ycbcr(1:2:t_r,1:2:t_c,2) = cb_idct_cpy(:,:);
rep_ycbcr(1:2:t_r,1:2:t_c,3) = cr_idct_cpy(:,:);
% temp var to hold upsampled cb and cr values
temp_cb = rep_ycbcr(:,:,2);
temp_cr = rep_ycbcr(:,:,3);

% performing linear interpolation on cb and cr components for upsampling
if mod(t_r,2) == 0
    for r = 2:2:t_r - 2
        for c = 1:2:t_c
            temp_cb(r,c) = round((temp_cb(r-1,c)+temp_cb(r+1,c))/2);
            temp_cr(r,c) = round((temp_cr(r-1,c)+temp_cr(r+1,c))/2);
        end
    end
    for c = 2:2:t_c - 2
        for r = 1:1:t_r
            temp_cb(r,c) = round((temp_cb(r,c-1)+temp_cb(r,c+1))/2);
            temp_cr(r,c) = round((temp_cr(r,c-1)+temp_cr(r,c+1))/2);
        end
    end
    for r = t_r
        for c = t_c
            temp_cb(r,c) = temp_cb(r-1,c);
            temp_cr(r,c) = temp_cr(r-1,c);
        end
    end
else
    for r = 2:2:t_r - 1
        for c = 1:2:t_c
            temp_cb(r,c) = round((temp_cb(r-1,c)+temp_cb(r+1,c))/2);
            temp_cr(r,c) = round((temp_cr(r-1,c)+temp_cr(r+1,c))/2);
        end
    end
    for c = 2:2:t_c - 1
        for r = 1:2:t_r
            temp_cb(r,c) = round((temp_cb(r,c-1)+temp_cb(r,c+1))/2);
            temp_cr(r,c) = round((temp_cr(r,c-1)+temp_cr(r,c+1))/2);
        end
    end
end

rep_ycbcr = cat(3,y,temp_cb,temp_cr);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: dc_ac_per_blk(frame,dc_y,ac_y,ac_r)
% Discription: Compute the Dc and AC(zig-zag) coefficients for the given
% 8x8 block of the frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dc_y,ac_y,dc_c,ac_c] = dc_ac_per_blk(frame,dc_y,ac_y,ac_r)
temp = size(zeros((144/8)*(176/8)*63),1);
temp2 = size(zeros((144/8)*(176/8)),1);
ac_c = 1;
dc_c = 1;
if (ac_r <= 5)
    temp_blk = zeros(8,8);
    [y_r,y_c] = size(frame);
    for r = 1:8:y_r
        for c = 1:8:y_c
            if ((r+7) <= y_r) && ((c+7) <= y_c)
                temp_blk = frame(r:r+7, c:c+7);
                temp = zigzag(temp_blk);
                ac_y(ac_r,ac_c:ac_c+62) = temp(2:end);
                dc_y(ac_r,dc_c:dc_c+7) = temp(1);
                ac_c = ac_c + 63;
                dc_c = dc_c + 1;
            end
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ::::::::::::::::::The Decoder::::::::::::::::::::
% Function: decoder(y,cb,cr,lum_q_matrix,chr_q_matrix)
% Discription: The decoder fucntion to assist with the decoder part of
% the video codec
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d_f_rgb = decoder(y,cb,cr,lum_q_matrix,chr_q_matrix)
[y_r,y_c] = size(y);
% i-DCT and i-qaunt on I frame
[iq_y,iq_cb,iq_cr,y_idct,cb_idct,cr_idct] = inv_dct_quant(y,cb,cr,lum_q_matrix,chr_q_matrix);
% performing linear interpolation on the I-frame.
[d_f] = lin_int_rc(y_idct,cb_idct,cr_idct,y_r,y_c);
% now converting the reconstructed ycbcr image to RGB
d_f_rgb = ycbcr2rgb(d_f);
end
