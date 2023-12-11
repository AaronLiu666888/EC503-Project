% Load your unbalanced data
load('bmi.mat'); 

% Separate features and labels
X = table2array(data_table(:, 2:3));
y = table2array(data_table(:, 4));

% Check class distribution
disp('Class distribution before balancing:');
disp(tabulate(y));

% Split data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training);
X_test = X(cv.test, :);
y_test = y(cv.test);

% Train initial model on the unbalanced data
initial_model = fitcknn(X_train, y_train, 'NumNeighbors', 5);

% Evaluate the initial model on the test set
y_initial_pred = predict(initial_model, X_test);

% Evaluate performance metrics for the initial model
initial_confusion_matrix = confusionmat(y_test, y_initial_pred);
initial_accuracy = sum(diag(initial_confusion_matrix)) / sum(initial_confusion_matrix(:));

% Compute precision, recall, and F1 score for the initial model
initial_precision = initial_confusion_matrix(2, 2) / sum(initial_confusion_matrix(:, 2));
initial_recall = initial_confusion_matrix(2, 2) / sum(initial_confusion_matrix(2, :));
initial_f1_score = 2 * (initial_precision * initial_recall) / (initial_precision + initial_recall);

%disp('Initial Model Metrics:');
disp(['Initial Accuracy: ', num2str(initial_accuracy)]);
%disp(['Initial Precision: ', num2str(initial_precision)]);
%disp(['Initial Recall: ', num2str(initial_recall)]);
disp(['Initial F1 Score: ', num2str(initial_f1_score)]);

% Identify minority class samples
minority_class = 1; % Change this based on your specific dataset
minority_indices = find(y_train == minority_class);

% Apply KNN SMOTE to balance the training data
k = 20; % Number of neighbors to consider
num_synthetic_samples = round(0.5 * sum(y_train == 1)); % You can adjust this ratio

synthetic_samples = zeros(num_synthetic_samples, size(X_train, 2));

for i = 1:num_synthetic_samples
    % Randomly select a minority class sample
    random_minority_index = minority_indices(randi(length(minority_indices)));
    minority_sample = X_train(random_minority_index, :);

    % Find k-nearest neighbors of the minority sample
    distances = pdist2(minority_sample, X_train);
    [~, sorted_indices] = sort(distances);
    nearest_neighbors = X_train(sorted_indices(2:k+1), :);

    % Generate a synthetic sample
    weights = rand(1, size(X_train, 2));
    synthetic_sample = minority_sample + weights .* (nearest_neighbors(randi(k), :) - minority_sample);

    % Add the synthetic sample to the dataset
    synthetic_samples(i, :) = synthetic_sample;
end

% Concatenate the synthetic samples with the original data
X_train_balanced = [X_train; synthetic_samples];
y_train_balanced = [y_train; ones(num_synthetic_samples, 1)];

% Check class distribution after balancing
disp('Class distribution after balancing:');
disp(tabulate(y_train_balanced));

% Train your model on the balanced data
balanced_model = fitcknn(X_train_balanced, y_train_balanced, 'NumNeighbors', 5);

% Evaluate the model on the test set
y_pred = predict(balanced_model, X_test);

% Evaluate performance metrics
confusion_matrix = confusionmat(y_test, y_pred);
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));

% Compute precision, recall, and F1 score
precision = confusion_matrix(2, 2) / sum(confusion_matrix(:, 2));
recall = confusion_matrix(2, 2) / sum(confusion_matrix(2, :));
f1_score = 2 * (precision * recall) / (precision + recall);

%disp('Balanced Model Metrics:');
disp(['Balanced Accuracy: ', num2str(accuracy)]);
%disp(['Balanced Precision: ', num2str(precision)]);
%disp(['Balanced Recall: ', num2str(recall)]);
disp(['Balanced F1 Score: ', num2str(f1_score)]);
