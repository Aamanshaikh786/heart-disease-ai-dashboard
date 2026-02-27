import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib 

df=pd.read_csv("heart.csv")
print("First 5 rows: \n",df.head())
print("\nShape: ",df.shape)
print("Columns: ",df.columns)
print("Missing Values: ",df.isnull().sum())
print("Target distribution: \n",df['target'].value_counts())

X=df.drop("target",axis=1)
Y=df["target"]
print("/n features Shape: ",X.shape)
print("Target Shape: ",Y.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("\nTraining Data: ",X_train.shape)
print("Testing Data: ",X_test.shape)

Scaler=StandardScaler()
X_train=Scaler.fit_transform(X_train)
X_test=Scaler.transform(X_test)

## LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr=lr.predict(X_test)

#Result
print("\n Logistic Regression===")
print("Accuracy = ",accuracy_score(Y_test,Y_pred_lr))
print("Confusion Matrix: \n",confusion_matrix(Y_test,Y_pred_lr))
print("Classification Report: \n",classification_report(Y_test,Y_pred_lr))


## Random forest
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
Y_pred_rf=rf.predict(X_test)

#Result
print("\n Random Forest ====")
print("Accuracy = ",accuracy_score(Y_test,Y_pred_rf))
print("Confusion Matrix: \n",confusion_matrix(Y_test,Y_pred_rf))
print("Classification Report: \n",classification_report(Y_test,Y_pred_rf))


## XGBoost (Advanced Model)
xgb=XGBClassifier(use_label_encoder=False,eval_metric='logloss')
xgb.fit(X_train,Y_train)
Y_pred_xgb=xgb.predict(X_test)

#Result
print("\n XGBoost ====")
print("Accuracy = ",accuracy_score(Y_test,Y_pred_xgb))
print("Confusion Matrix: \n",confusion_matrix(Y_test,Y_pred_xgb))
print("Classification Report: \n",classification_report(Y_test,Y_pred_xgb))


# ROC Curve

y_prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
auc_score = roc_auc_score(Y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="Random Forest (AUC = %.2f)" % auc_score)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", auc_score)


# Feature Importance
importance = rf.feature_importances_
feature_names = df.drop("target", axis=1).columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df)


# save model 
joblib.dump(rf, "model.pkl")

# Save scaler
joblib.dump(Scaler, "scaler.pkl")

print("Model and Scaler Saved Successfully!")