# Credit_Risk_Analysis

## Overview
For this analysis, our purpose is to analyze credit risk from data generated from LendingClub, a peer-to-peer lending services company. Using the scikit-learn and imbalanced-learn libraries, we are evaluating models using resampling. We are also oversampling the data using RandomOverSampler and SMOTE algorithms, as well as undersampling the data using the ClusterCentroids algorithm. Then we combine these approaches by using the SMOTEENN algorithm. And finally, to reduce bias, we are using two machine learning models: BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Results
### Oversample Model
![Oversample-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146698547-12c8272b-6cd7-4e09-a611-d2c62a338aa9.png)
- With the oversample model, we can see that our Balanced Accuracy Score is 64.7%.

![Oversample-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146698619-dab11b82-5ed7-41ff-893d-14f16eccb4d3.png)
![Oversample-ClassificationReport](https://user-images.githubusercontent.com/69607218/146698665-56b93bf1-95a9-4e05-9bc2-e459291e6045.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, a recall (sensitivity) of 62%, and a F1 score 2%.
- Our 'low_risk' precision is very high at 100%, with a recall (sensitivity) of 67%, and our F1 score of 80%.

### Undersample Model
![Undersample-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146698962-e6465b28-1658-4263-b24c-edceafb4208f.png)
- With the undersample model, we can see that our Balanced Accuracy Score is 51.77%.

![Undersample-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146699022-fb586a9f-c2bf-41d3-bfeb-1261f7862c62.png)
![Undersample-ClassificationReport](https://user-images.githubusercontent.com/69607218/146699024-96d65343-3c19-449d-b378-4b5479a1ad23.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, a recall (sensitivity) of 57%, and a F1 score of 1%.
- Our 'low_risk' precision is very high again at a 100%, with a recall (sensitivity) of 46%, and a F1 score of 63%.

### SMOTE Model
![SMOTE-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146699302-610923e3-5838-4983-b22e-92bd0617bfe7.png)
- With the SMOTE model, we can see that our Balanced Accuracy Score is 62.5%.

![SMOTE-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146699349-6fcf7525-fbc0-48d1-a69c-16b1dd4e82c0.png)
![SMOTE-ClassificationReport](https://user-images.githubusercontent.com/69607218/146699350-71ee85b4-fd7f-445f-8bf9-02c81db27628.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, a recall (sensitivity) of 62%, and a F1 score of 2%.
- Our 'low_risk' precision is again 100%, with a recall (sensitivity) of 63%, and a F1 score of 77%.

### SMOTEENN Model
![SMOTEENN-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146699548-3011a7f9-c6fd-434e-a276-66c189ec97ef.png)
- With the SMOTEENN model, we can see that our Balanced Accuracy Score is 62.5% (the same as the SMOTE model).

![SMOTEENN-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146699597-f7e654bb-63cf-4ffa-9953-26f8dddbd9ec.png)
![SMOTEENN-ClassificationReport](https://user-images.githubusercontent.com/69607218/146699605-cb40847f-cd71-499f-ab51-ac9864ed48eb.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 1%, with a recall (sensitivity) of 71%, and a F1 score of 2%.
- Our 'low_risk' precision is again 100%, with a recall (sensitivity) of 54%, and a F1 score of 70%.

### Balanced Random Forest Classifier Model
![RandomForest-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146699884-8889bda0-c7d8-4f27-be84-fc74af7675fb.png)
- With the Balanced Random Forest Classifier model, we can see that our Balanced Accuracy Score is 78.78%.

![RandomForest-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146699935-f515e2a2-9dca-473d-bc49-3fe0f67ab9c3.png)
![RandomForest-ClassificationReport](https://user-images.githubusercontent.com/69607218/146699938-d77e945a-0700-4147-b050-2638ca535b3c.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 4%, with a recall (sensitivity) of 67%, and a F1 score of 7%.
- Our 'low_risk' precision is 100%, with a recall (sensitivity) of 91%, and a F1 score of 95%.

### Easy Ensemble Classifier Model
![EasyEnsemble-BalancedAccuracyScore](https://user-images.githubusercontent.com/69607218/146702765-92539f10-f5c4-4bf5-94f1-0bf2441c4a09.png)
- With the Easy Ensemble Classifier model, we can see that our Balanced Accuracy Score is at 92.5%.

![EasyEnsemble-ConfusionMatrix](https://user-images.githubusercontent.com/69607218/146702909-7a5a480e-499b-4e46-8b20-e9a75f3d385f.png)
![EasyEnsemble-ClassificationReport](https://user-images.githubusercontent.com/69607218/146702926-f72c17d1-5fb0-4c4f-8b94-2c9fa8496ece.png)
- Using our Confusion Matrix to calculate our Imbalanced Classification Report, we can see that our 'high_risk' precision is at 7%, with a recall (sensitivity) of 91%, and a F1 score of 14%.
- Our 'low_risk' precision is 100%, with a recall (sensitivity) of 94%, and a F1 score of 97%.

## Summary
All of the models above show a weak precision to determining if a credit risk is high. Out of the 6 models, the Easy Ensemble model performed the best with an accuracy score of 92.5%. As well with a recall score of 91% for 'high_risk' and a 94% for 'low_risk', this provides the best and most accurate information amongst the models. That being said, I would recommend that the Easy Ensemble Classifier model is the best to choose from when determining credit risk for the company LendingClub.
