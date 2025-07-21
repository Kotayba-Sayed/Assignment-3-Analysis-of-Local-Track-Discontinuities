# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# %%
df1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
df2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
df3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")

# %%
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

columns_to_drop = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
combined_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# %%
combined_df['event'] = combined_df['event'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

# %%
X = combined_df.drop(columns=['event'])
y = combined_df['event']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# %%
X_scaled_df['event'] = y
X_scaled_df.to_csv("preprocessed_dataset.csv", index=False)

print("Preprocessing complete.")
print("Shape of data:", X_scaled_df.shape)
print("Event class distribution:\n", y.value_counts())
