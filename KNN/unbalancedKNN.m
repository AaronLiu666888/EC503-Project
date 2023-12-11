% bmi.mat -> just the bmi.csv to mat loaded in
load('bmi.mat'); 

% Separate features and labels
X = table2array(data_table(:, 2:3));
y = table2array(data_table(:, 4));

% Original Class Distribution
disp('Class distribution before balancing:');
disp(tabulate(y));

% Test & Training sets
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training);
X_test = X(cv.test, :);
y_test = y(cv.test);

% Train based on original
initial_model = fitcknn(X_train, y_train, 'NumNeighbors', 5);

% Predict Original
y_initial_pred = predict(initial_model, X_test);

% Calculate Accuracy Original
initial_confusion_matrix = confusionmat(y_test, y_initial_pred);
initial_accuracy = sum(diag(initial_confusion_matrix)) / sum(initial_confusion_matrix(:));

% Calculate F1 Score Original
initial_precision = initial_confusion_matrix(2, 2) / sum(initial_confusion_matrix(:, 2));
initial_recall = initial_confusion_matrix(2, 2) / sum(initial_confusion_matrix(2, :));
initial_f1_score = 2 * (initial_precision * initial_recall) / (initial_precision + initial_recall);

disp(['Initial Accuracy: ', num2str(initial_accuracy)]);
disp(['Initial F1 Score: ', num2str(initial_f1_score)]);

% Undersampled classes
minority_class = 1; 
minority_indices = find(y_train == minority_class);

% Apply KNN SMOTE 
k = 5; % CHANGE
num_synthetic_samples = round(0.5 * sum(y_train == 1)); % number of samples

synthetic_samples = zeros(num_synthetic_samples, size(X_train, 2));

for i = 1:num_synthetic_samples
    % Random selection
    random_minority_index = minority_indices(randi(length(minority_indices)));
    minority_sample = X_train(random_minority_index, :);

    % KNN of minority
    distances = pdist2(minority_sample, X_train);
    [~, sorted_indices] = sort(distances);
    nearest_neighbors = X_train(sorted_indices(2:k+1), :);

    % create sample
    weights = rand(1, size(X_train, 2));
    synthetic_sample = minority_sample + weights .* (nearest_neighbors(randi(k), :) - minority_sample);

    % Add sample to array of samples
    synthetic_samples(i, :) = synthetic_sample;
end

% Combine original with new
X_train_balanced = [X_train; synthetic_samples];
y_train_balanced = [y_train; ones(num_synthetic_samples, 1)];

% Balanced Class Distriution
disp('Class distribution after balancing:');
disp(tabulate(y_train_balanced));

% Train based on balanced data
balanced_model = fitcknn(X_train_balanced, y_train_balanced, 'NumNeighbors', 5);

% Predict Balanced
y_pred = predict(balanced_model, X_test);

% Calculate Accuracy
confusion_matrix = confusionmat(y_test, y_pred);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));

% Calculate F1 Score
precision = confusion_matrix(2, 2) / sum(confusion_matrix(:, 2));
recall = confusion_matrix(2, 2) / sum(confusion_matrix(2, :));
f1_score = 2 * (precision * recall) / (precision + recall);

disp(['Balanced Accuracy: ', num2str(accuracy)]);
disp(['Balanced F1 Score: ', num2str(f1_score)]);
