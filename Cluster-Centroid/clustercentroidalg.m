clear all; close all; clc; 
rawdata = readmatrix("Wine_Quality.xlsx"); %imbalanced dataset
% for simplicity - let the number of feature vectors = 2
data_imbalanced = rawdata(:,3:4);
labels_imbalanced = rawdata(:,2);
%note that this dataset is imbalanced - the goal of this clustering algorithm is to
%even out the imbalance by randomly removing points from the class with the
%larger number of data points
clear rawdata;

% chose initial centroid for majority class
ind1 = find(labels_imbalanced(labels_imbalanced==1)); ind2 = find(labels_imbalanced(labels_imbalanced==2)); %finding which indexes correspond w each class
if length(ind1)>length(ind2)
    majorityclass = "class #1"; 
    majoritydata = data_imbalanced(ind1,:); minoritydata = data_imbalanced(ind2,:);
else length(ind1)<length(ind2);
    majorityclass = "class #2";
    majoritydata = data_imbalanced(ind2,:); minoritydata = data_imbalanced(ind1,:);
end 
centroid = mean(majoritydata,1,"omitmissing"); % initial centroid for majority class
% storing initial values in structure for easy access
important.initialmajoritycentroid = centroid;
important.initialsizemajority = size(majoritydata);
important.initialsizeminority = size(minoritydata);
clear ind1 ind2;

% plot dataset before processing 
figure
subplot(1,2,1)
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
labels_balanced = labels_imbalanced;
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
    labels_balanced(eliminate) = [];
    centroid = mean(majoritydata,1,"omitmissing");  %update majority centroid
    clear i; clear ans; clear distances; clear threshold; clear eliminate; 
end 
% storing final values in structure for easy access
important.finalmajoritycentroid = centroid;
important.finalsizemajority = size(majoritydata);
important.finalsizeminority = size(minoritydata);
clear numeliminate abovethreshold belowthreshold;

% re-plot datasets after random elimination of majority class
subplot(1,2,2)
hold all 
plot(majoritydata(:,1),majoritydata(:,2),'bo','LineWidth',1,'MarkerSize',3)
plot(minoritydata(:,1),minoritydata(:,2),'mo','LineWidth',1,'MarkerSize',3)
plot(centroid(1),centroid(2),'k^','LineWidth',2,'MarkerSize',5)
grid minor 
xlabel('Feature #1')
ylabel('Feature #2')
title('Plotting Dataset After Running Algorithm')
legend('Majority','Minority','Centroid','Location','northeast'); 

% combine newly-formed clusters together into 1 dataset 
data_balanced = [majoritydata;minoritydata]; %save this 
important.balanceddata = data_balanced; 
clear majoritydata minoritydata; 
