
# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load datasets
df1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
df2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
df3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")

# %% combine datasets
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

columns_to_drop = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
combined_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# %% Encode 'event' column: normal → 0, others → 1
combined_df['event'] = combined_df['event'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

# %% Feature/label separation and normalization
X = combined_df.drop(columns=['event'])
y = combined_df['event']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# %% Save preprocessed dataset
X_scaled_df['event'] = y
X_scaled_df.to_csv("preprocessed_dataset.csv", index=False)

print("Preprocessing complete.")
print("Shape of data:", X_scaled_df.shape)
print("Event class distribution:\n", y.value_counts())

# %% SVM Model Training and Evaluation
X = X_scaled_df.drop(columns=["event"])
y = X_scaled_df["event"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

svm_model_split = SVC(kernel='rbf', C=1.0)
svm_model_split.fit(X_train, y_train)
y_pred_split = svm_model_split.predict(X_test)

print("\n=== SVM Performance: 80/20 Train-Test Split ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_split))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_split))

# %% 5-Fold Cross-Validation
svm_model_cv = SVC(kernel='rbf', C=1.0)
cv_scores = cross_val_score(svm_model_cv, X, y, cv=5, scoring='accuracy')

print("=== SVM Performance: 5-Fold Cross-Validation ===")
print("Fold Accuracies:", cv_scores)
print("Mean Accuracy: {:.4f}".format(cv_scores.mean()))
print("Standard Deviation: {:.4f}".format(cv_scores.std()))

# %% Confusion Matrix for 80/20 split
cm = confusion_matrix(y_test, y_pred_split)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Event"],
            yticklabels=["Normal", "Event"])
plt.title("Confusion Matrix - 80/20 Split")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
