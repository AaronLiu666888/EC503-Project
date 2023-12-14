Team 2: Some Classical ML Methods for Unbalanced & Missing Data

Team Members: Kristie Gong, Aaron Liu, Tanvi Shingade

Github Link: https://github.com/AaronLiu666888/EC503-Project

Original Data Sources: 

	Missing Datasets:
 
		Wine Quality (All 3 Algorithms):
  
			https://www.kaggle.com/datasets/ilayaraja07/data-cleaning-feature-imputation
   
	Unbalanced Datasets:
 
		Wine Quality (Cluster Centroid Algorithm):
  
			https://www.kaggle.com/datasets/ilayaraja07/data-cleaning-feature-imputation
   
		BMI (K-Nearest Neighbors Algorithm):
  
			https://www.kaggle.com/datasets/yasserh/bmidataset
   
		Student Performance (K-Means Algorithm):
  
		https://www.kaggle.com/datasets/ilayaraja07/data-cleaning-feature-imputation 

Dataset Cleaning/Processing:

Cluster Centroid

	For both missing and imbalanced datasets, the Wine Quality dataset was used. 
 
	The number of feature vectors was limited to 2 (K=2), and the entirety of the dataset was loaded into MATLAB. 
 
	This dataset primarily presented as imbalanced, with the number of data points skewed at a 75:25 ratio between classes. 
 
	Some data points were missing certain values, which is why no additional missing dataset was used. The classification of data points was determined based on Wine Type (White or Red, where White was the majority and therefore classified as Class 1), and the classes were added as a separate column to the original Excel Spreadsheet to simplify the MATLAB code.
 	
  	This edited Excel Spreadsheet (found in the Cluster Centroid folder on GitHub) was then loaded into MATLAB for the CC algorithm. 
   
K-Nearest Neighbors

	Wine Quality Dataset
		This was cleaned/processed by replacing the categorical data with numerical values and turning the .csv file into a .mat file. There is only one feature that is categorical which is “type”. White was set to 1 and Red was set to 0. This is the WQ_Modified.mat.
  
		For the instance of testing, all missing value rows were removed from the WQ_Modified.mat to produce WQ_Actual.mat. WQ_Test.mat was also created for testing which had random removal of data from WQ_Actual.mat.
  
		For baseline understanding, WQ_Modified.mat ‘s missing values were filled with the medians of the columns resulting in the MedianFill.mat.
  
	BMI Dataset
 
		This dataset did not need any cleaning or processing besides turning the .csv file to a .mat file.
K-Means
Wine Quality Dataset
The preprocessing of this dataset is to impute the missing value with the mean of the numeric column
The features (independent variables) and the label (dependent variable, 'quality') are separated.
Categorical variables are converted into numerical values using one-hot encoding. 
Numerical features are normalized
Students Performance Dataset
It segregates the data into two groups based on the 'testPreparationCourse' column.
To address imbalance, it randomly selects samples from the larger group to match the size of the smaller group.
The script extracts numerical features (math, reading, and writing scores) and standardizes them.

Original Supporting Code Used:
Cluster Centroid
Did not use any major functions that computed components of the CC algorithm.
K-Nearest Neighbors
Some MATLAB functions were used for efficiency of running through data. 
i.e. fitcknn()
i.e. knnimpute()
K-Means
Some MATLAB functions were used for efficiency of running through data. 
i.e. strcmp()
i.e. randperm()
i.e. pca()

Codes Explanation:
Cluster Centroid
Running the code normally (F5) will produce the exact same results that I included in the final report. Note that this code will produce less than 5 figures, and should take less than 1min to finish running completely. If you notice some variables mentioned in the code that aren’t visible in the workspace - don’t worry. Those are only used for computing other variables/values and were cleared at the end of the code. Check the command window for critical outputs. 

K-Nearest Neighbors
unbalancedKNN.m
Loaded data can be changed to desired .mat file
Currently set to bmi.mat
Separates array to label and features
bmi.mat: 
Features: height & weight Label: BMI
Displays original distribution of classes 
Trains model to calculate accuracy and F1 score of original distribution
Accuracy: (Correct Fills) / (Total Fills)
F1 Score: (2 * (Initial Precision) * (Initial Recall)) / ((Initial Precision) + (Initial Recall))
Uses k-nearest neighbors to create synthetic data points to include in the class distribution
Creates new dataset which includes the synthetic data points
Displays new distribution of classes
Trains model to calculate accuracy and F1 score of new distribution
Accuracy: (Correct Fills) / (Total Fills)
F1 Score: (2 * (Initial Precision) * (Initial Recall)) / ((Initial Precision) + (Initial Recall))
missingKNN.m (Actual Fill Result)
Fills missing data with k-nearest neighbors
Change the loaded data to desired .mat file
This one is set to WQ_Modified.mat.
Line 13: Change k to a number of your choice
Used k = 5, 10, 20
Saves filled data .mat to WQ_Modified_Filled.mat
missingKNNtesting.m (Test Fill Result)
Fills missing data with k-nearest neighbors
Change the loaded data to desired .mat file 
This one is set to WQ_Test.mat.
Line 11: Change k to a number of your choice
Used k = 5, 10, 20
Saves filled data .mat to WQ_Test_Filled#.mat
# refers to the k value selected if various k value .mat files are desired.
This can be applied to missingKNN.m.
This is essentially the same as missingKNN.m.
missingKNNtestingcompare.m (Comparing Result to Actual)
Compares the actual dataset and the test filled dataset
Calculates the accuracy of the missing values
(Correct Fills) / (Total Fills)
Calculates the mean absolute error of the fill
Average of the absolute difference between all fills
Calculates the root square mean error of the fill
Square root of the average of differences between all fills

K-Means

missingdata_Kmeans.m:
	
 	Loading the Dataset
 
	Handling Missing Values in Numeric Columns
 
	Updating the Dataset
 
	Separating Features and Label
 
	Categorical to Numerical Conversion (One-Hot Encoding)
 
	Normalization and PCA
 
	Applying K-means Clustering
 
	Silhouette Score Calculation
 
imbalanced_Kmeans.m:

	Loading the Dataset
 
	Segregating the Data
 
	Balancing the Dataset
 
	Combining Subsets for a Balanced Dataset
 
	Feature Standardization
 
	Applying PCA and Plotting
 
	K-means Clustering Implementation
 
	Centroid Calculation and Plotting
 
	Silhouette Score for Clustering Evaluation

Notes to Reproduce Results:

Cluster Centroid

	N/A - check the above section for details. 

K-Nearest Neighbors

	Missing Data

		The dataset tends to perform better with sorted .mat files.

K-Means

	N/A - check the above section for details. 
