%% Notes
% this program 
% 1) preprocesses the data provided by the organization committee 
% 2) prepares data for time series classification
% 3) visualizes some data series for presentation.
% by Zhiming Zhang et al. (see the paper for the full author list)
%% Remove Outlier
% this section removes the outlier in some data series
% details can be found in the paper and presentation video
clear;close all;clc
load('CF2007-12-14.mat')
% data series with outlier
dataTemp = CF.Data(:,6);
% build the threshold for outlier detection
maxCF = max(dataTemp(22000:30000));
minCF = min(dataTemp(22000:30000));
% remove outlier
dataTemp(dataTemp>maxCF) = minCF;
dataTemp(dataTemp<minCF) = minCF;
CF.Data(:,6) = dataTemp;
% save cleaned data
save('CF2007-12-14-Modified.mat','CF')
%%
clear;close all;clc
load('CF2009-05-05.mat')
% data series with outlier
dataTemp = CF.Data(:,6);
% build the threshold for outlier detection
maxCF = max(dataTemp(22000:30000));
minCF = min(dataTemp(22000:30000));
% remove outlier
dataTemp(dataTemp>maxCF) = minCF;
dataTemp(dataTemp<minCF) = minCF;
CF.Data(:,6) = dataTemp;
% save cleaned data
save('CF2009-05-05-Modified.mat','CF')
%%
clear;close all;clc
load('CF2011-11-01.mat')
% data series with outlier
dataTemp = CF.Data(:,6);
% build the threshold for outlier detection
maxCF = max(dataTemp(22000:30000));
minCF = min(dataTemp(22000:30000));
% remove outlier
dataTemp(dataTemp>maxCF) = minCF;
dataTemp(dataTemp<minCF) = minCF;
CF.Data(:,6) = dataTemp;
% data series with outlier
dataTemp = CF.Data(:,13);
% build the threshold for outlier detection
maxCF = max(dataTemp(22000:30000));
minCF = min(dataTemp(22000:30000));
% remove outlier
dataTemp(dataTemp>maxCF+20) = minCF;
CF.Data(:,13) = dataTemp;
% save cleaned data
save('CF2011-11-01-Modified.mat','CF')
%% Data Visualization
% load data
clear;close all;clc
dataRec(10) = struct();
j = 1;
for i = 13:19
    str = strcat('CF2006-05-',num2str(i),'.mat');
    load(str);dataRec(j).data = CF.Data;j = j+1;
end
load('CF2007-12-14-Modified.mat')
dataRec(j).data = CF.Data;j = j+1;
load('CF2009-05-05-Modified.mat')
dataRec(j).data = CF.Data;j = j+1;
load('CF2011-11-01-Modified.mat')
dataRec(j).data = CF.Data;
%%
% plot cable force for each cable
close all
set(0,'DefaultAxesTitleFontWeight','normal');
f = fig('units','inches','width',10,'height',2,'font','Times New Roman','fontsize',11);
h = tight_subplot(2,5,[.06 .04],[.04 .04],[.04 .01]);
j = 1;  % 1,2,...14
for i=1:10
    dataT = dataRec(i).data;
    axes(h(i));box on;
    plot(dataT(:,j),'b')
    set(gca,'xtick',[]);
    textT = strcat('(',char(i+'a'-1),')');
    xl = xlim;yl = ylim;
    xt = xl(1)+0.85*(xl(2)-xl(1));
    yt = yl(1)+0.9*(yl(2)-yl(1));
    text(xt,yt,textT)
end
% saveas(f,'FigureTemp/cf_SJS8.png')
% saveas(f,'FigureTemp/cf_SJS8.eps','epsc') 
% savefig(f,'FigureTemp/cf_SJS8.fig')
%%
% plot cable force ratio for each cable pair
close all
set(0,'DefaultAxesTitleFontWeight','normal');
f = fig('units','inches','width',10,'height',2,'font','Times New Roman','fontsize',11);
h = tight_subplot(2,5,[.06 .04],[.04 .04],[.04 .01]);
j = 1;  % 1,2,...7
for i=1:10
    dataT = dataRec(i).data;
    axes(h(i));box on;
    plot(dataT(:,j)./dataT(:,j+7),'b')
    set(gca,'xtick',[]);
    textT = strcat('(',char(i+'a'-1),')');
    xl = xlim;yl = ylim;
    xt = xl(1)+0.85*(xl(2)-xl(1));
    yt = yl(1)+0.9*(yl(2)-yl(1));
    text(xt,yt,textT)
end
% saveas(f,'FigureTemp/cfr_SJ8.png')
% saveas(f,'FigureTemp/cfr_SJ8.eps','epsc') 
% savefig(f,'FigureTemp/cfr_SJ8.fig')
%% Data Preparation: cable force (Scenario 1)
% this section prepares data for TSC with LSTM-FCN model
clear;close all;clc
dataLen = 1600; % length of time series each segment
% load and reshape each dataset
dataTemp = [];
for i = 13:19
    str = strcat('CF2006-05-',num2str(i),'.mat');
    load(str);
    dataTemp = [dataTemp;reshape(CF.Data,dataLen,[])']; 
end
load('CF2007-12-14-Modified.mat')
dataTemp = [dataTemp;reshape(CF.Data,dataLen,[])']; 
load('CF2009-05-05-Modified.mat')
dataTemp = [dataTemp;reshape(CF.Data,dataLen,[])']; 
load('CF2011-11-01-Modified.mat')
data2011 = reshape(CF.Data,dataLen,[])'; % data in 2011

dataLen0 = size(CF.Data,1); % length of original series
numSens = size(CF.Data,2);  % number of sensors, number of classes
numDates = 10;              % number of days with collected data
labels = [];                % labels for the time series samples
for i=1:numSens, labels = [labels;i*ones(dataLen0/dataLen,1)];end
labels2011 = labels;        % labels for data in 2011
data2011 = [labels data2011];   % time seris data in 2011 with labels
% data with labels in 2011 for each cable
% the last 13 rows are the artificial data for other classes
% which is necessary to use the LSTM-FCN TSC model
data2011CableSJS8 = [data2011(1:108,:);repmat(mean(data2011),13,1)];
data2011CableSJS9 = [data2011(108*1+1:108*2,:);repmat(mean(data2011),13,1)];
data2011CableSJS10 = [data2011(108*2+1:108*3,:);repmat(mean(data2011),13,1)];
data2011CableSJS11 = [data2011(108*3+1:108*4,:);repmat(mean(data2011),13,1)];
data2011CableSJS12 = [data2011(108*4+1:108*5,:);repmat(mean(data2011),13,1)];
data2011CableSJS13 = [data2011(108*5+1:108*6,:);repmat(mean(data2011),13,1)];
data2011CableSJS14 = [data2011(108*6+1:108*7,:);repmat(mean(data2011),13,1)];
data2011CableSJX8 = [data2011(108*7+1:108*8,:);repmat(mean(data2011),13,1)];
data2011CableSJX9 = [data2011(108*8+1:108*9,:);repmat(mean(data2011),13,1)];
data2011CableSJX10 = [data2011(108*9+1:108*10,:);repmat(mean(data2011),13,1)];
data2011CableSJX11 = [data2011(108*10+1:108*11,:);repmat(mean(data2011),13,1)];
data2011CableSJX12 = [data2011(108*11+1:108*12,:);repmat(mean(data2011),13,1)];
data2011CableSJX13 = [data2011(108*12+1:108*13,:);repmat(mean(data2011),13,1)];
data2011CableSJX14 = [data2011(108*13+1:108*14,:);repmat(mean(data2011),13,1)];
data2011CableSJS8(end-12:end,1) = setdiff(1:14,1)';
data2011CableSJS9(end-12:end,1) = setdiff(1:14,2)';
data2011CableSJS10(end-12:end,1) = setdiff(1:14,3)';
data2011CableSJS11(end-12:end,1) = setdiff(1:14,4)';
data2011CableSJS12(end-12:end,1) = setdiff(1:14,5)';
data2011CableSJS13(end-12:end,1) = setdiff(1:14,6)';
data2011CableSJS14(end-12:end,1) = setdiff(1:14,7)';
data2011CableSJX8(end-12:end,1) = setdiff(1:14,8)';
data2011CableSJX9(end-12:end,1) = setdiff(1:14,9)';
data2011CableSJX10(end-12:end,1) = setdiff(1:14,10)';
data2011CableSJX11(end-12:end,1) = setdiff(1:14,11)';
data2011CableSJX12(end-12:end,1) = setdiff(1:14,12)';
data2011CableSJX13(end-12:end,1) = setdiff(1:14,13)';
data2011CableSJX14(end-12:end,1) = setdiff(1:14,14)';
% all data in 2010 with labels
labels = repmat(labels,numDates-1,1);
data2010 = [labels dataTemp];
% separate databefore 2011 to test and train equally
lenTrn= round(size(data2010,1)/2);
trnInd = randperm(size(data2010,1),lenTrn);
testInd = setdiff(1:size(data2010,1),trnInd);
data2010Trn = data2010(trnInd,:);
data2010Test = data2010(testInd,:);
% write data to file
dlmwrite('cfPre2011_TRAIN.tsv',data2010Trn,'delimiter','\t')
dlmwrite('cfPre2011_TEST.tsv',data2010Test,'delimiter','\t')
dlmwrite('cf2011.tsv',data2011,'delimiter','\t')
dlmwrite('cf2011SJS8.tsv',data2011CableSJS8,'delimiter','\t')
dlmwrite('cf2011SJS9.tsv',data2011CableSJS9,'delimiter','\t')
dlmwrite('cf2011SJS10.tsv',data2011CableSJS10,'delimiter','\t')
dlmwrite('cf2011SJS11.tsv',data2011CableSJS11,'delimiter','\t')
dlmwrite('cf2011SJS12.tsv',data2011CableSJS12,'delimiter','\t')
dlmwrite('cf2011SJS13.tsv',data2011CableSJS13,'delimiter','\t')
dlmwrite('cf2011SJS14.tsv',data2011CableSJS14,'delimiter','\t')
dlmwrite('cf2011SJX8.tsv',data2011CableSJX8,'delimiter','\t')
dlmwrite('cf2011SJX9.tsv',data2011CableSJX9,'delimiter','\t')
dlmwrite('cf2011SJX10.tsv',data2011CableSJX10,'delimiter','\t')
dlmwrite('cf2011SJX11.tsv',data2011CableSJX11,'delimiter','\t')
dlmwrite('cf2011SJX12.tsv',data2011CableSJX12,'delimiter','\t')
dlmwrite('cf2011SJX13.tsv',data2011CableSJX13,'delimiter','\t')
dlmwrite('cf2011SJX14.tsv',data2011CableSJX14,'delimiter','\t')
%% Data Preparation: cable force ratio (Scenario 2)
% see the section above for more comments
clear;close all;clc
dataLen = 1600;
dataTemp = [];
for i = 13:19
    str = strcat('CF2006-05-',num2str(i),'.mat');
    load(str);
    forceR = zeros(size(CF.Data,1),size(CF.Data,2)/2);
    for j=1:size(forceR,2)
        forceR(:,j) = CF.Data(:,j)./CF.Data(:,j+size(forceR,2));
    end
    dataTemp = [dataTemp;reshape(forceR,dataLen,[])']; 
end
load('CF2007-12-14-Modified.mat')
forceR = zeros(size(CF.Data,1),size(CF.Data,2)/2);
for j=1:size(forceR,2)
    forceR(:,j) = CF.Data(:,j)./CF.Data(:,j+size(forceR,2));
end
dataTemp = [dataTemp;reshape(forceR,dataLen,[])']; 
load('CF2009-05-05-Modified.mat')
forceR = zeros(size(CF.Data,1),size(CF.Data,2)/2);
for j=1:size(forceR,2)
    forceR(:,j) = CF.Data(:,j)./CF.Data(:,j+size(forceR,2));
end
dataTemp = [dataTemp;reshape(forceR,dataLen,[])']; 
load('CF2011-11-01-Modified.mat')
forceR = zeros(size(CF.Data,1),size(CF.Data,2)/2);
for j=1:size(forceR,2)
    forceR(:,j) = CF.Data(:,j)./CF.Data(:,j+size(forceR,2));
end
data2011 = reshape(forceR,dataLen,[])'; 

dataLen0 = size(CF.Data,1);
numSens = size(CF.Data,2);
numDates = 10;      
labels = [];
for i=1:numSens/2,labels = [labels;i*ones(dataLen0/dataLen,1)];end
labels2011 = labels;
data2011 = [labels data2011];
labels = repmat(labels,numDates-1,1);
data2010 = [labels dataTemp];

% data with labels in 2011 for each cable pair
% sensors on SJ8 and SJ13 are broken
data2011SJ9 = data2011(108+1:108*2,:);
data2011SJ9 = [data2011SJ9;repmat(mean(data2011SJ9),6,1)];
data2011SJ9(end-5:end,1) = [1 3:7]';
%
data2011SJ10 = data2011(108*2+1:108*3,:);
data2011SJ10 = [data2011SJ10;repmat(mean(data2011SJ10),6,1)];
data2011SJ10(end-5:end,1) = [1:2 4:7]';
%
data2011SJ11 = data2011(108*3+1:108*4,:);
data2011SJ11 = [data2011SJ11;repmat(mean(data2011SJ11),6,1)];
data2011SJ11(end-5:end,1) = [1:3 5:7]';
%
data2011SJ12 = data2011(108*4+1:108*5,:);
data2011SJ12 = [data2011SJ12;repmat(mean(data2011SJ12),6,1)];
data2011SJ12(end-5:end,1) = [1:4 6:7]';
%
data2011SJ14 = data2011(108*6+1:108*7,:);
data2011SJ14 = [data2011SJ14;repmat(mean(data2011SJ14),6,1)];
data2011SJ14(end-5:end,1) = [1:6]';

% separate data before 2011 to test and train equally
lenTrn= round(size(data2010,1)/2);
trnInd = randperm(size(data2010,1),lenTrn);
testInd = setdiff(1:size(data2010,1),trnInd);
data2010Trn = data2010(trnInd,:);
data2010Test = data2010(testInd,:);

% write data to file
dlmwrite('cfrPre2011_TRAIN.tsv',data2010Trn,'delimiter','\t')
dlmwrite('cfrPre2011_TEST.tsv',data2010Test,'delimiter','\t')
dlmwrite('cfr2011.tsv',data2011,'delimiter','\t')
dlmwrite('cfr2011SJ9.tsv',data2011SJ9,'delimiter','\t')
dlmwrite('cfr2011SJ10.tsv',data2011SJ10,'delimiter','\t')
dlmwrite('cfr2011SJ11.tsv',data2011SJ11,'delimiter','\t')
dlmwrite('cfr2011SJ12.tsv',data2011SJ12,'delimiter','\t')
dlmwrite('cfr2011SJ14.tsv',data2011SJ14,'delimiter','\t')