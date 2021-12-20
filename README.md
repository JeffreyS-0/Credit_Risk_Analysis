# Credit_Risk_Analysis

## Overview
For this analysis, our purpose is to analyze credit risk from data generated from LendingClub, a peer-to-peer lending services company. Using the scikit-learn and imbalanced-learn libraries, we are evaluating models using resampling. We are also oversampling the data using RandomOverSampler and SMOTE algorithms, as well as undersampling the data using the ClusterCentroids algorithm. Then we combine these approaches by using the SMOTEENN algorithm. And finally, to reduce bias, we are using two machine learning models: BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Results
### Oversample Model
![Oversample-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146698547-12c8272b-6cd7-4e09-a611-d2c62a338aa9.png)
- With the oversample model, we can see that our Balanced Accuracy Score is 64.7%.

![Oversample-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146698619-dab11b82-5ed7-41ff-893d-14f16eccb4d3.png)
![Oversample-ClassificationReport](https://user-images.githubusercontent.com/69607218/146698665-56b93bf1-95a9-4e05-9bc2-e459291e6045.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, a recall (sensitivity) of 62%, and a F-1 score 2%.
- Our 'low_risk' precision is very high at 100%, with a recall (sensitivity) of 67%, and our F-1 score of 80%.

### Undersample Model
![Undersample-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146698962-e6465b28-1658-4263-b24c-edceafb4208f.png)
- With the undersample model, we can see that our Balanced Accuracy Score is 51.77%.

![Undersample-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146699022-fb586a9f-c2bf-41d3-bfeb-1261f7862c62.png)
![Undersample-ClassificationReport](https://user-images.githubusercontent.com/69607218/146699024-96d65343-3c19-449d-b378-4b5479a1ad23.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, a recall (sensitivity) of 57%, and a F-1 score of 1%.
- Our 'low_risk' precision is very high again at a 100%, with a recall (sensitivity) of 46%, and a F-1 score of 63%.

### SMOTE Model
