load('WQ_Actual.mat');

% Actual Correct
actualX = data(:, 1:end-1);
actualY = data(:, end);


load('WQ_Test_Filled20.mat');

% Assumed by KNN
imputedX = data(:, 1:end-1);
imputedY = data(:, end);

totaldatanum = length(data);

counter = 0;

if abs(imputedX(missingIndices) - actualX(missingIndices)) < 0.1
    counter = counter + 1;
end

accuracy = counter/totaldatanum;

% MAE and RMSE
mae = mean(abs(imputedX(missingIndices) - actualX(missingIndices)));
rmse = sqrt(mean((imputedX(missingIndices) - actualX(missingIndices)).^2));

disp(['Accuracy: ', num2str(accuracy)]);
disp(['Mean Absolute Error (MAE): ', num2str(mae)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(rmse)]);

