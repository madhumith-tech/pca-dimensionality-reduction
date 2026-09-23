import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

data = pd.read_csv(
    "optdigits.tra",
    header=None
)

print("=" * 60)
print("1. DATASET LOADING")
print("=" * 60)

print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 2. EXPLORE THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("2. DATASET EXPLORATION")
print("=" * 60)

print("\nDataset information:")
data.info()

print("\nStatistical summary:")
print(data.describe())

print("\nNumber of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\n" + "=" * 60)
print("3. FEATURES AND TARGET")
print("=" * 60)

print("Number of features:", X.shape[1])
print("Number of target columns:", 1)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget values:")
print(sorted(y.unique()))


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("4. MISSING VALUE CHECK")
print("=" * 60)

missing_values = X.isnull().sum().sum()

print("Total missing values:", missing_values)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("5. HANDLING MISSING VALUES")
print("=" * 60)

imputer = SimpleImputer(strategy="mean")

X_imputed = imputer.fit_transform(X)

print("Missing values after preprocessing:",
      np.isnan(X_imputed).sum())


# ============================================================
# 6. STANDARDIZE NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("6. STANDARDIZATION")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Standardized data shape:", X_scaled.shape)

print("\nMean after standardization:")
print(round(X_scaled.mean(), 4))

print("\nStandard deviation after standardization:")
print(round(X_scaled.std(), 4))


# ============================================================
# 7. APPLY PCA
# ============================================================

print("\n" + "=" * 60)
print("7. PCA")
print("=" * 60)

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

print("PCA transformed shape:", X_pca.shape)


# ============================================================
# 8. EXPLAINED VARIANCE
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\n" + "=" * 60)
print("8. EXPLAINED VARIANCE")
print("=" * 60)

for i, variance in enumerate(explained_variance, start=1):
    print(
        "PC",
        i,
        ":",
        round(variance * 100, 2),
        "%"
    )


# ============================================================
# 9. CUMULATIVE EXPLAINED VARIANCE
# ============================================================

cumulative_variance = np.cumsum(
    explained_variance
)

print("\n" + "=" * 60)
print("9. CUMULATIVE EXPLAINED VARIANCE")
print("=" * 60)

for i, variance in enumerate(
    cumulative_variance,
    start=1
):
    print(
        "PC",
        i,
        ":",
        round(variance * 100, 2),
        "%"
    )


# ============================================================
# 10. PLOT EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

plt.title(
    "Explained Variance by Principal Component"
)

plt.grid()

plt.show()


# ============================================================
# 11. PLOT CUMULATIVE EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title(
    "Cumulative Explained Variance"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 12. DETERMINE NUMBER OF COMPONENTS
# ============================================================

n_components = np.argmax(
    cumulative_variance >= 0.95
) + 1

print("\n" + "=" * 60)
print("10. NUMBER OF COMPONENTS")
print("=" * 60)

print(
    "Components required to retain 95% variance:",
    n_components
)

print(
    "Variance retained:",
    round(
        cumulative_variance[n_components - 1] * 100,
        2
    ),
    "%"
)


# ============================================================
# 13. TRANSFORM DATA USING SELECTED PCA COMPONENTS
# ============================================================

pca_reduced = PCA(
    n_components=n_components
)

X_reduced = pca_reduced.fit_transform(
    X_scaled
)

print("\n" + "=" * 60)
print("11. PCA TRANSFORMATION")
print("=" * 60)

print(
    "Original data shape:",
    X.shape
)

print(
    "Reduced data shape:",
    X_reduced.shape
)


# ============================================================
# 14. COMPARE ORIGINAL AND REDUCED DIMENSIONS
# ============================================================

original_dimensions = X.shape[1]

reduced_dimensions = X_reduced.shape[1]

dimension_reduction = (
    (original_dimensions - reduced_dimensions)
    / original_dimensions
) * 100

print("\n" + "=" * 60)
print("12. DIMENSION COMPARISON")
print("=" * 60)

print(
    "Original dimensions:",
    original_dimensions
)

print(
    "Reduced dimensions:",
    reduced_dimensions
)

print(
    "Dimensions reduced by:",
    round(dimension_reduction, 2),
    "%"
)


# ============================================================
# 15. VISUALIZE REDUCED-DIMENSIONAL DATA
# ============================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=y,
    cmap="tab10",
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "PCA Reduced-Dimensional Data"
)

plt.colorbar(
    scatter,
    label="Digit"
)

plt.grid()

plt.show()


# ============================================================
# 16. CREATE REDUCED DATASET
# ============================================================

pca_columns = []

for i in range(
    1,
    n_components + 1
):
    pca_columns.append(
        "PC" + str(i)
    )


pca_data = pd.DataFrame(
    X_reduced,
    columns=pca_columns
)

pca_data["Digit"] = y.values


# ============================================================
# 17. DISPLAY REDUCED DATA
# ============================================================

print("\n" + "=" * 60)
print("13. REDUCED DATASET")
print("=" * 60)

print("\nFirst 5 rows:")
print(pca_data.head())


# ============================================================
# 18. SAVE REDUCED DATASET
# ============================================================

pca_data.to_csv(
    "pca_reduced_data.csv",
    index=False
)

print("\nReduced dataset saved as:")
print("pca_reduced_data.csv")


# ============================================================
# 19. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("14. FINAL INTERPRETATION")
print("=" * 60)

print(
    "PCA reduced the original",
    original_dimensions,
    "features to",
    reduced_dimensions,
    "principal components."
)

print(
    "The selected components retain approximately",
    round(
        cumulative_variance[n_components - 1] * 100,
        2
    ),
    "% of the total variance."
)

print(
    "This reduces the dimensionality of the dataset "
    "while preserving most of its important information."
)

print(
    "The first two principal components were used "
    "to visualize the reduced data."
)