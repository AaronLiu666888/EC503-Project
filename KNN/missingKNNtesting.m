load('WQ_Test.mat');  % type white = 1 red = 0 + random missing data

% Find missing indexes
missingIndices = isnan(data);

% features and labels
X = data(:, 1:end-1);  % Features
y = data(:, end);      % Labels

% KNN to find fill
k = 20;  % You can adjust the value of k as needed
imputedX = knnimpute(X, k);

% Mix changes
imputedData = [imputedX, y];

% Replace missing with knn
data(missingIndices) = imputedData(missingIndices);

% Just Temporary Graphing
gscatter(data(:, 10), data(:, 9), data(:, 1));
xlabel('pH')
ylabel('density')

% Save to New .mat file with filled data
save('WQ_Test_Filled20.mat', 'data');
