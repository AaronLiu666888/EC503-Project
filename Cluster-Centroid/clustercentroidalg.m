clear all; close all; clc; 
rawdata = readmatrix("Wine_Quality.xlsx"); %imbalanced dataset
% for simplicity - let the number of feature vectors = 2
data_pre = rawdata(:,3:4);
labels_pre = rawdata(:,2);
%note that this dataset is imbalanced - the goal of this clustering algorithm is to
%even out the imbalance by randomly removing points from the class with the
%larger number of data points

% chose initial centroid for majority class
ind1 = find(labels_pre(labels_pre==1)); ind2 = find(labels_pre(labels_pre==2)); %finding which indexes correspond w each class
if length(ind1)>length(ind2)
    majorityclass = "class #1"; 
    majoritydata = data_pre(ind1,:); minoritydata = data_pre(ind2,:);
else length(ind1)<length(ind2);
    majorityclass = "class #2";
    majoritydata = data_pre(ind2,:); minoritydata = data_pre(ind1,:);
end 
centroid = mean(majoritydata,1,"omitmissing"); % initial centroid for majority class
% storing initial values in structure for easy access
predata.majoritycentroid = centroid;
predata.sizemajority = size(majoritydata);
predata.sizeminority = size(minoritydata);
predata.mean = [mean(data_pre(:,1),1,"omitmissing"),mean(data_pre(:,2),1,"omitmissing")];
predata.variance = [var(data_pre(:,1),1,"omitmissing"),var(data_pre(:,2),1,"omitmissing")];
predata.range = [range(data_pre(:,1),1),range(data_pre(:,2),1)];
disp("critical components of data before processing:")
disp(predata)

% plot dataset before processing 
figure
% subplot(1,2,1)
hold all 
plot(majoritydata(:,1),majoritydata(:,2),'bo','LineWidth',1,'MarkerSize',3)
plot(minoritydata(:,1),minoritydata(:,2),'mo','LineWidth',1,'MarkerSize',3)
plot(centroid(1),centroid(2),'k^','LineWidth',2,'MarkerSize',5)
grid minor 
xlabel('Feature #1')
ylabel('Feature #2')
title('Plotting Dataset Before Running Algorithm')
legend('Majority','Minority','Centroid','Location','northeast'); 

% WHILE LOOP: calculate distances + eliminate points + update centroid
numeliminate = 25;
labels_post = labels_pre;
while length(majoritydata)>=length(minoritydata)
    distances = zeros(length(majoritydata),1);
    for i=1:length(majoritydata)
        distances(i,:) = norm(majoritydata(i,:)-centroid);
    end 
    threshold = mean(distances,"omitmissing");
    belowthreshold = find(distances(distances<threshold));
    abovethreshold = find(distances(distances>threshold));
    eliminate = [randsample(belowthreshold,numeliminate);randsample(abovethreshold,numeliminate)];
    majoritydata(eliminate,:) = [];
    labels_post(eliminate) = [];
    centroid = mean(majoritydata,1,"omitmissing");  %update majority centroid
end 
% combine newly-formed clusters together into 1 dataset 
data_post = [majoritydata;minoritydata];

% storing final values in structure for easy access
postdata.majoritycentroid = centroid;
postdata.sizemajority = size(majoritydata);
postdata.sizeminority = size(minoritydata);
postdata.mean = [mean(data_post(:,1),1,"omitmissing"),mean(data_post(:,2),1,"omitmissing")];
postdata.variance = [var(data_post(:,1),1,"omitmissing"),var(data_post(:,2),1,"omitmissing")];
postdata.range = [range(data_post(:,1),1),range(data_post(:,2),1)];
disp("critical components of data after processing:")
disp(postdata)

% re-plot datasets after random elimination of majority class
figure
% subplot(1,2,2)
hold all 
plot(majoritydata(:,1),majoritydata(:,2),'bo','LineWidth',1,'MarkerSize',3)
plot(minoritydata(:,1),minoritydata(:,2),'mo','LineWidth',1,'MarkerSize',3)
plot(centroid(1),centroid(2),'k^','LineWidth',2,'MarkerSize',5)
grid minor 
xlabel('Feature #1')
ylabel('Feature #2')
title('Plotting Dataset After Running Algorithm')
legend('Majority','Minority','Centroid','Location','northeast'); 

% check to see if pre+post data retain the mean,variance,range,spread
figure
hold all
plot(data_pre(:,1),data_pre(:,2),'bo','LineWidth',.5,'MarkerSize',3)
plot(data_post(:,1),data_post(:,2),'bo','LineWidth',1.5,'MarkerSize',3)
grid minor 
xlabel('Feature #1')
ylabel('Feature #2')
title('Comparing Majority Cluster Before+After Processing')
legend('Majority before Processing','Majority after Processing','Location','northeast'); 

% structure compiling differences between the pre and post data
comparison.means = (abs(predata.mean-postdata.mean)./(predata.mean+postdata.mean))*100;
comparison.variances = (abs(predata.variance-postdata.variance)./(predata.variance+postdata.variance))*100;
comparison.ranges = (abs(predata.range-postdata.range)./(predata.range+postdata.range))*100;
disp("% differences in data before and after processing:")
disp(comparison)

clear rawdata ind1 ind2 numeliminate abovethreshold belowthreshold i ans distances threshold eliminate majoritydata minoritydata; 