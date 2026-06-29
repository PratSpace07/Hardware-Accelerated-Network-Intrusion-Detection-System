!pip install pandas numpy scikit-learn -q

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


# Load the Thursday Infiltration CSV
file_name = "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
df = pd.read_csv(file_name)

# Clean column names: remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Show the first few rows and columns to verify
print("Columns:", df.columns.tolist())
df.head()


# Load the CSV (adjust filename if needed)
file_name = "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
df = pd.read_csv(file_name)

# Clean column names
df.columns = df.columns.str.strip()

# Features chosen for FPGA implementation
FEATURES = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "SYN Flag Count",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Mean",
    "Flow IAT Mean",
    "Destination Port"
]

# ------------------------------------------------------------
# 1. Filter only BENIGN traffic
benign_df = df[df["Label"] == "BENIGN"].copy()

# 2. Extract feature matrix
X = benign_df[FEATURES].values

# 3. Convert to float (just to be safe)
X = X.astype(np.float64)

# 4. Remove rows that contain NaN or inf
#    Use np.isnan() and np.isinf() to find bad rows
bad_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
print(f"Total benign rows: {X.shape[0]}")
print(f"Rows with NaN or inf: {bad_mask.sum()}")
X_clean = X[~bad_mask]
print(f"Clean rows left: {X_clean.shape[0]}")

if X_clean.shape[0] == 0:
    raise ValueError("No clean benign rows left! Check your data.")

# 5. Fit scaler and PCA on the clean data
scaler = StandardScaler().fit(X_clean)
Xs = scaler.transform(X_clean)

# 6. Fit PCA (start with 4 components; you can adjust after seeing explained variance)
N_COMPONENTS = 4
pca = PCA(n_components=N_COMPONENTS).fit(Xs)

# 7. (Optional) Check reconstruction error on the training set to compute threshold later
Y = pca.transform(Xs)
X_hat = pca.inverse_transform(Y)

print("✅ Scaler and PCA fitted successfully on clean benign data.")
print(f"Number of benign samples used: {X_clean.shape[0]}")
print(f"Mean of scaled features: {Xs.mean(axis=0)}")
print(f"Std of scaled features: {Xs.std(axis=0)}")



# Cell 1b: Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative = np.cumsum(explained_variance)

print("Explained variance per component:", explained_variance)
print("Cumulative explained variance:", cumulative)
print(f"Total explained variance with {N_COMPONENTS} components: {cumulative[-1]:.4f}")

# If you want to automatically choose the number of components to reach 95%, uncomment:
# N_COMPONENTS = np.argmax(cumulative >= 0.95) + 1
# print(f"Recommended number of components for 95% variance: {N_COMPONENTS}")


# Compute reconstruction error for benign data
recon_err = np.sum((Xs - X_hat) ** 2, axis=1)
threshold = np.percentile(recon_err, 99)   # tune this for desired false alarm rate

print(f"Reconstruction error threshold (99th percentile): {threshold:.6f}")

# ------------------------------------------------------------
# Fixed‑point conversion parameters
FRAC_BITS = 12
TOTAL_BITS = 16

def to_fixed(val: float) -> int:
    """Convert float to Q4.12 fixed‑point (signed, two's complement)."""
    scaled = int(round(val * (1 << FRAC_BITS)))
    # Mask to TOTAL_BITS bits (two's complement wrap)
    return scaled & ((1 << TOTAL_BITS) - 1)

def write_mem(path, values):
    """Write a list of floats to a .mem file (one hex value per line)."""
    with open(path, "w") as f:
        for v in values:
            f.write(f"{to_fixed(v):04x}\n")

# Export mean, invstd, PCA components (flattened row‑major), and threshold
write_mem("mean.mem", scaler.mean_)
write_mem("invstd.mem", 1.0 / scaler.scale_)   # pre‑computed reciprocal std
write_mem("pca_w.mem", pca.components_.flatten())  # shape (N_COMP, N_FEAT) row‑major
write_mem("threshold.mem", [threshold])

print("✅ Fixed‑point .mem files generated:")
for f in ["mean.mem", "invstd.mem", "pca_w.mem", "threshold.mem"]:
    print(f"  - {f}")

# (Optional) Also generate stimulus.mem and labels for simulation
# This uses the same CSV but you can also use a separate test file
# For demonstration, we'll create stimulus from all data (benign + attack)
# but you might want to use a different held‑out file.

# Uncomment to create a stimulus file from the whole dataset (for simulation)
X_all = scaler.transform(df[FEATURES].values)   # use the same scaler
write_mem("stimulus.mem", X_all.flatten())
df[["Label"]].to_csv("expected_labels.csv", index=False)
print("  - stimulus.mem and expected_labels.csv generated.")



# ------------------------------------------------------------
# Generate stimulus.mem and expected_labels.csv for simulation
# (cleans the entire dataset before scaling)
# ------------------------------------------------------------

# 1. Extract features from the FULL dataset
X_all_raw = df[FEATURES].values.astype(np.float64)

# 2. Remove rows that contain NaN or inf (from BOTH benign and attacks)
bad_mask = np.isnan(X_all_raw).any(axis=1) | np.isinf(X_all_raw).any(axis=1)
print(f"Total rows in dataset: {X_all_raw.shape[0]}")
print(f"Rows with NaN or inf: {bad_mask.sum()}")

# 3. Keep only clean rows
X_all_clean = X_all_raw[~bad_mask]
labels_clean = df["Label"].values[~bad_mask]   # corresponding labels

print(f"Clean rows left: {X_all_clean.shape[0]}")

# 4. Scale using the already-fitted scaler (from benign training)
X_scaled = scaler.transform(X_all_clean)

# 5. Write stimulus.mem (flattened, one feature per line)
write_mem("stimulus.mem", X_scaled.flatten())

# 6. Write expected labels for scoring after simulation
pd.DataFrame(labels_clean, columns=["Label"]).to_csv("expected_labels.csv", index=False)

print("✅ Stimulus files generated:")
print(f"  - stimulus.mem  ({X_scaled.shape[0]} vectors, {X_scaled.shape[1]} features each)")
print(f"  - expected_labels.csv  ({len(labels_clean)} labels)")
