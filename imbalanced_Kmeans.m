% Load the data
data = readtable('Students_Performance_knn.csv');

% Separate the data into two groups based on 'testPreparationCourse'
data_none = data(strcmp(data.testPreparationCourse, 'none'), :);
data_completed = data(strcmp(data.testPreparationCourse, 'completed'), :);

% Randomly select samples from the larger group to match the size of the smaller group
num_samples = min(height(data_none), height(data_completed));
data_none_subset = data_none(randperm(height(data_none), num_samples), :);
data_completed_subset = data_completed(randperm(height(data_completed), num_samples), :);

% Combine these subsets to create a new balanced dataset
balanced_data = [data_none_subset; data_completed_subset];

% Extract numerical data for clustering
features = balanced_data{:, {'mathScore', 'readingScore', 'writingScore'}};

% Standardize the features
features_scaled = zscore(features);

% Apply K-means clustering
K = 3; % Number of clusters
maxIter = 100; % Maximum iterations
[clusterIdx, centroids] = KMeans(features_scaled, K, maxIter);

silhouette(features_scaled, clusterIdx);

% Optionally, to get the average silhouette score
silh_vals = silhouette(features_scaled, clusterIdx);
avg_silh_score = mean(silh_vals);
fprintf('Average Silhouette Score: %f\n', avg_silh_score);

% Applying PCA for visualization
[coeff, score, ~, ~, explained] = pca(features_scaled);
reducedData = score(:, 1:2); % Taking first two principal components

% Plotting the clusters
figure;
gscatter(reducedData(:,1), reducedData(:,2), clusterIdx);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-means Clustering with PCA');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');


function [clusterIdx, centroids] = KMeans(data, K, maxIter)
    % CustomKMeans performs K-means clustering on the dataset
    % Inputs:
    %   data - A matrix of data points
    %   K - Number of clusters
    %   maxIter - Maximum number of iterations

    % Randomly initialize the centroids
    centroids = data(randperm(size(data, 1), K), :);
    oldCentroids = zeros(size(centroids));
    clusterIdx = zeros(size(data, 1), 1);
    iter = 0;

    while ~isequal(centroids, oldCentroids) && iter < maxIter
        oldCentroids = centroids;

        % Assign data points to the nearest centroid
        for i = 1:size(data, 1)
            [~, clusterIdx(i)] = min(sum((data(i, :) - centroids).^2, 2));
        end

        % Update centroids
        for j = 1:K
            centroids(j, :) = mean(data(clusterIdx == j, :), 1);
        end

        iter = iter + 1;
    end
end

