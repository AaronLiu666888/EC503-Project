% Load the dataset
data = readtable('Students_Performance_knn.csv');

% Separate features and label
labels = data.gender;  % Label
features = data(:, 2:end);  % Features excluding 'gender'

% Convert categorical variables to numerical (one-hot encoding)
categoricalFeatures = {'race_ethnicity', 'parentalLevelOfEducation', 'lunch', 'testPreparationCourse'};
for i = 1:length(categoricalFeatures)
    % Create dummy variables
    dummies = dummyvar(categorical(features.(categoricalFeatures{i})));
    
    % Generate unique column names for the dummy variables
    dummyNames = strcat(categoricalFeatures{i}, '_', string(1:size(dummies, 2)));
    
    % Add the dummy variables to the dataset
    features = [features array2table(dummies, 'VariableNames', dummyNames)];
    
    % Remove the original categorical column
    features.(categoricalFeatures{i}) = [];
end

% Normalize numerical scores
numericalFeatures = {'mathScore', 'readingScore', 'writingScore'};
features{:, numericalFeatures} = normalize(features{:, numericalFeatures});

% Visualization using gscatter
% Using 'mathScore' and 'readingScore' for visualization
gscatter(features.mathScore, features.readingScore, labels);
title('Original Dataset');
xlabel('Math Score');
ylabel('Reading Score');

% Convert categorical labels to numeric if they aren't already
[numericLabels, ~, labelIndices] = unique(labels);

% Split the dataset into training and testing sets
cv = cvpartition(size(features, 1), 'HoldOut', 0.3);
idx = cv.test;

% Separate training and testing sets
XTrain = table2array(features(~idx,:));
YTrain = labelIndices(~idx);
XTest = table2array(features(idx,:));

YTest = labelIndices(idx);

% Define K and class weights
K = 5; % Example value, adjust based on your dataset
classWeights = [1; 1.5]; % Example weights, adjust based on your dataset

% Apply KNN
predictedLabels = KNN(XTrain, YTrain, XTest, K, classWeights);

% Evaluate the model
% Convert predicted labels back to original categorical format
predictedLabelsCategorical = numericLabels(predictedLabels);

% Calculate accuracy or other metrics
accuracy = sum(strcmp(predictedLabelsCategorical, labels(idx))) / numel(predictedLabelsCategorical);
disp(['Accuracy: ', num2str(accuracy)]);

function predictedLabels = KNN(trainData, trainLabels, testData, K, classWeights)
    numTestPoints = size(testData, 1);
    numClasses = length(unique(trainLabels));
    predictedLabels = zeros(numTestPoints, 1);

    if nargin < 5
        % If class weights are not provided, use equal weights for all classes
        classWeights = ones(numClasses, 1);
    end

    for i = 1:numTestPoints
        % Calculate Euclidean distances (ensure trainData and testData are arrays)
        distances = sqrt(sum((trainData - testData(i, :)).^2, 2));

        % Sort distances (distances should be an array here)
        [sortedDistances, sortedIndices] = sort(distances);

        nearestIndices = sortedIndices(1:K);
        nearestDistances = sortedDistances(1:K);
        nearestLabels = trainLabels(nearestIndices);

        % Calculate weighted votes
        voteCounts = zeros(numClasses, 1);
        for j = 1:K
            label = nearestLabels(j);
            distanceWeight = 1 / (nearestDistances(j) + eps); % Add 'eps' to avoid division by zero
            voteCounts(label) = voteCounts(label) + distanceWeight * classWeights(label);
        end

        % Predict the label with the highest weighted vote
        [~, predictedLabels(i)] = max(voteCounts);
    end
end
