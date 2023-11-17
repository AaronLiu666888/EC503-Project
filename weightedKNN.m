function predictedLabels = weightedKNN(trainData, trainLabels, testData, K, classWeights)
    numTestPoints = size(testData, 1);
    numClasses = length(unique(trainLabels));
    predictedLabels = zeros(numTestPoints, 1);

    if nargin < 5
        % If class weights are not provided, use equal weights for all classes
        classWeights = ones(numClasses, 1);
    end

    for i = 1:numTestPoints
        % Calculate Euclidean distances
        distances = sqrt(sum((trainData - testData(i, :)).^2, 2));

        % Find K nearest neighbors
        [~, sortedIndices] = sort(distances);
        nearestIndices = sortedIndices(1:K);
        nearestDistances = distances(nearestIndices);
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
