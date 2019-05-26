clear;
% I choose the logistic classifier to be further tested for grading
folder = '~/MatlabProjects/cs567hw4/distributed';
testfolder = './';
filePattern = fullfile(folder, '*.ppm');
imgs = dir(filePattern);
filePattern = fullfile(testfolder, '*.ppm');
testimgs = dir(filePattern);
if ~isfolder(folder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', folder);
  uiwait(warndlg(errorMessage));
  return;
end

if ~isfolder(testfolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', testFolder);
  uiwait(warndlg(errorMessage));
  return;
end


for i = 1:length(imgs)
    baseFileName = imgs(i).name;
    fullFileName = fullfile(folder, baseFileName);
    oimg = imread(fullFileName);
    img = double(oimg);
    % *********************************************************************
    % Feature 1 - Number of bright spots
    % *********************************************************************
    % Step 1 Isolate green channel
    rawimg = 0*img(:,:,1)+1*img(:,:,2)+0*img(:,:,3); 
    % Step 2-1 Locate blood vessels in the image
    bh = imbothat(rawimg, strel('disk',10));
    % Step 2-2 Add blood vessels image to the original to lower visibility
    bhafter = rawimg+bh;
    % Step 3 Dilate image to enhance brighter areas
    se = strel('disk',25);
    dimg = imdilate(bhafter,se);
    % Step 4-1 Calculate threshold base on sorted brightness 
    V = sort(dimg(:), 'descend');
    top2 = V(ceil(end/10)*.2);
    % Step 4-2 Set threshold to only show top 2%
    dimg(dimg<top2)=0;
    % Step 5-1 Count the holes in dilated image. 
    % Step 5-2 One of them is probably the optic disc. Exclude it.
    feat1(i) = extractfield(bwconncomp(dimg),'NumObjects')-1;
    
    % *********************************************************************
    % Feature 2 - Number of dark points
    % *********************************************************************
    dimg2 =  bhafter;
    % Step 1 Using the image without blood vessel to extract feature 2
    V = sort(dimg2(:), 'ascend');
    % Step 2 Set threshold to only show top 55%
    dimg2 = dimg2>V(ceil(end/10*5));
    % Step 3 Erosion to enhance dark spots in the white blob
    dimg2 = imerode(dimg2,strel('disk',10));
    % Step 4-1 Count the black spots on the white blobs
    n=bwconncomp(dimg2);
    h=regionprops(n,'Eulernumber');
    % Step 4-2 Since it returns 1 - number of black spots, the blob with
    % most black spot is what we want. It is the minimum EulerNumber in the 
    % image. Number of black spots = 1 - minimum of the Euler Number.
    % Healthy retina could either show 1 black spot (Novea) or
    % none(neglected). So both -1 and 0 in the graph yields 0 haemmorage.
    feat2(i)=-1*min(extractfield(h,'EulerNumber'));
    feat2(feat2==-1) = 0;
    subplot(4,9,i);imagesc(rawimg);
    colormap(gray)
end
l = length(imgs);
feat = [feat1',feat2'];
f1= repmat(1:6, 1,l/6);
f1 = f1(randperm(l/2));
f2 = repmat(1:6, 1,l/6);
f2 = f2(randperm(l/2));

f = [f1, f2];

k = 9;
pred = zeros(length(imgs),1);
labels = ones(length(imgs),1);
labels(length(imgs)/2+1:length(imgs)) = 2;
for i=1:6
    train = feat(f~=i, :);
    test = feat(f==i, :);
    labels_train = labels(f ~= i);
    nfeat = size(train, 2);
    for n=1:nfeat
       mn_train = mean(train(:,n));
       sd_train = std(train(:,n));
       train(:,n) = (train(:,n)-mn_train)/sd_train;
       test(:,n) = (test(:,n)-mn_train)/sd_train;
    end
	ntrain = size(train, 1);
	ntest = size(test, 1);
    pred_test = zeros(1, ntest);
    
    for j=1:ntest
        dist = sqrt(sum((ones(ntrain,1)*test(j,:)-train).^2, 2));
        [reord, ord] = sort(dist);
        knn=labels_train(ord(1:k));
        p_g1 = mean(knn == 1);
        p_g2 = mean(knn == 2);
        if (p_g2<p_g1)
            pred_test(j)=1;
        elseif (p_g1<p_g2)
            pred_test(j)=2;
        else
            pred_test(j)=floor(rand()*2); 
        end   
    end
    pred(f == i) = pred_test;
end

match = labels == pred;
accuracy_g1 = mean(match(labels == 1));
accuracy_g2 = mean(match(labels == 2));
disp("Healthy retina using KNN is "+accuracy_g1);
disp("Unhealthy retina using KNN is "+accuracy_g2);

% TEST USING LOGISTIC
for i = 1:length(testimgs)
    baseFileName = testimgs(i).name;
    fullFileName = fullfile(folder, baseFileName);
    oimg = imread(fullFileName);
    img = double(oimg);
    % *********************************************************************
    % Feature 1 - Number of bright spots
    % *********************************************************************
    % Step 1 Isolate green channel
    rawimg = 0*img(:,:,1)+1*img(:,:,2)+0*img(:,:,3); 
    % Step 2-1 Locate blood vessels in the image
    bh = imbothat(rawimg, strel('disk',10));
    % Step 2-2 Add blood vessels image to the original to lower visibility
    bhafter = rawimg+bh;
    % Step 3 Dilate image to enhance brighter areas
    se = strel('disk',25);
    dimg = imdilate(bhafter,se);
    % Step 4-1 Calculate threshold base on sorted brightness 
    V = sort(dimg(:), 'descend');
    top2 = V(ceil(end/10)*.2);
    % Step 4-2 Set threshold to only show top 2%
    dimg(dimg<top2)=0;
    % Step 5-1 Count the holes in dilated image. 
    % Step 5-2 One of them is probably the optic disc. Exclude it.
    feat1(i) = extractfield(bwconncomp(dimg),'NumObjects')-1;
    
    % *********************************************************************
    % Feature 2 - Number of dark points
    % *********************************************************************
    dimg2 =  bhafter;
    % Step 1 Using the image without blood vessel to extract feature 2
    V = sort(dimg2(:), 'ascend');
    % Step 2 Set threshold to only show top 55%
    dimg2 = dimg2>V(ceil(end/10*5));
    % Step 3 Erosion to enhance dark spots in the white blob
    dimg2 = imerode(dimg2,strel('disk',10));
    % Step 4-1 Count the black spots on the white blobs
    n=bwconncomp(dimg2);
    h=regionprops(n,'Eulernumber');
    % Step 4-2 Since it returns 1 - number of black spots, the blob with
    % most black spotis what we want. It is the minimum EulerNumber in the 
    % image. Number of black spots = 1 - minimum of the Euler Number.
    % Healthy retina could either show 1 black spot (Novea) or
    % none(neglected). So both -1 and 0 in the graph yields 0 haemmorage.
    feat2(i)=-1*min(extractfield(h,'EulerNumber'));
    feat2(feat2==-1) = 0;
    subplot(4,9,i);imagesc(rawimg);
    colormap(gray)
end
l = length(testimgs);
feat = [feat1',feat2'];
labels = labels-1;
        
folds1= repmat(1:6, 1,3);
folds1 = folds1(randperm(18));
folds2 = repmat(1:6, 1,3);
folds2 = folds2(randperm(18));

folds = [folds1, folds2];

pred = zeros(size(labels));
    
for i=1:6
    train = feat(folds~=i, :);
    test = feat(folds==i, :);
    labels_train = labels(folds ~= i);
    nfeat = size(train, 2);
    for n=1:nfeat
       mn_train = mean(train(:,n));
       sd_train = std(train(:,n));
       train(:,n) = (train(:,n)-mn_train)/sd_train;
       test(:,n) = (test(:,n)-mn_train)/sd_train;
    end
	ntest = size(test, 1);
	ntrain = size(train, 1);
	pred_test = zeros(1, ntest);
    
    beta = glmfit(train, labels_train, 'binomial', 'link', 'logit');
  
    xb = [ones(size(test,1), 1), test]*beta;
    prob_test = exp(xb)./(1+exp(xb));
    pred_test = 1*prob_test>.5;
  
    pred(folds == i) = pred_test;
end
match = labels == pred;
accuracy_g1 = mean(match(labels == 0));
accuracy_g2 = mean(match(labels == 1));
disp("Healthy retina using Logistic is "+accuracy_g1);
disp("Unhealthy retina using Logistic is "+accuracy_g2);


healthy = feat2;
score = sum(healthy(1:length(imgs)/2)==0)+sum(healthy(19:36)~=0);
score