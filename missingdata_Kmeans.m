% Load the dataset
data = readtable('Wine_Quality.csv', 'VariableNamingRule', 'preserve');

% Handling missing values for numeric columns only
numericCols = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
numericData = data(:, numericCols);

% Calculate the mean for each numeric column, ignoring NaNs
numericMeans = zeros(1, width(numericData));
for i = 1:width(numericData)
    numericMeans(i) = mean(numericData{:, i}, 'omitnan');
end

% Filling missing values with the mean of each numeric column
for i = 1:width(numericData)
    numericData{:, i} = fillmissing(numericData{:, i}, 'constant', numericMeans(i));
end

% Replace the numeric columns in the original dataset with the filled columns
data(:, numericCols) = numericData;

% Separate features and label
labels = data.quality;
features = data(:, 1:end-1);

% Convert categorical variables to numerical (one-hot encoding)
categoricalFeatures = {'type'};
for i = 1:length(categoricalFeatures)
    % Create dummy variables
    dummies = dummyvar(grp2idx(data.(categoricalFeatures{i})));
    
    % Generate unique column names for the dummy variables
    dummyNames = strcat(categoricalFeatures{i}, '_', string(1:size(dummies, 2)));
    
    % Add the dummy variables to the dataset
    features = [features(:, setdiff(features.Properties.VariableNames, categoricalFeatures{i})) array2table(dummies, 'VariableNames', dummyNames)];    
end

% Normalize numerical features
numericalFeatures = setdiff(features.Properties.VariableNames, categoricalFeatures);
features{:, numericalFeatures} = normalize(features{:, numericalFeatures});

% Convert the features table to a numeric matrix
features_matrix = table2array(features);

% Dimensionality Reduction using PCA
[coeff1, score, ~, ~, ~] = pca(features_matrix);
reducedData = score(:, 1:2);

% Plotting the data before clustering
figure;
scatter(reducedData(:,1), reducedData(:,2));
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Data Visualization Before K-means Clustering');

% Apply K-means clustering
K = 2;
maxIter = 100;
[clusterIdx, centroids] = KMeans(features_matrix, K, maxIter);

% Dimensionality Reduction using PCA
[coeff, score, ~, ~, ~] = pca(features_matrix);
reducedData = score(:, 1:2);

% Plotting the clusters
figure;
gscatter(reducedData(:,1), reducedData(:,2), clusterIdx);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-means Clustering with PCA Reduction');
legend(arrayfun(@(x) ['Cluster ' num2str(x)], 1:K, 'UniformOutput', false));

% Convert centroids to the same PCA-reduced space
centroidsReduced = centroids * coeff(:,1:2);
hold on;
plot(centroidsReduced(:,1), centroidsReduced(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
hold off;

% Calculate the silhouette score
silh_vals = silhouette(features_matrix, clusterIdx);
avg_silh_score = mean(silh_vals);
fprintf('Average Silhouette Score: %f\n', avg_silh_score);

function [clusterIdx, centroids] = KMeans(data, K, maxIter)

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


