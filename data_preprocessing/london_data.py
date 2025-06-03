import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
import os
num_users = 500


# ----------------------------------------------------------------------
# 0. READ + NORMALISE COLUMN NAMES
# ----------------------------------------------------------------------
# full data
data_path = Path("./data/london")
csv_file  = data_path / "CC_LCL-FullData.csv"

df = (pd.read_csv(csv_file, low_memory=False)
        .rename(columns=lambda c: c.strip()))                 # ← trims blanks

top_k_lclid = df['LCLid'].unique()[:num_users]
df = df[df['LCLid'].isin(top_k_lclid)]

# <---- add this line
df["KWH/hh (per half hour)"] = pd.to_numeric(
        df["KWH/hh (per half hour)"], errors="coerce")

df = df[["LCLid", "DateTime", "KWH/hh (per half hour)"]]
df["DateTime"] = pd.to_datetime(df["DateTime"])
# ----------------------------------------------------------------------
# 1.  EXTRA TIME KEYS
# ----------------------------------------------------------------------
print("extra time keys.")
df["Date"]       = df["DateTime"].dt.date
df["slot"]       = df["DateTime"].dt.hour*2 + df["DateTime"].dt.minute//30
df["Month"]      = df["DateTime"].dt.month
df["DayOfWeek"]  = df["DateTime"].dt.dayofweek
df["DayOfMonth"] = df["DateTime"].dt.day
df["WeekOfYear"] = df["DateTime"].dt.isocalendar().week.astype(int)

# handle NaN
# 1️⃣  Make sure the energy column is numeric (coerce errors → NaN)
df['KWH/hh (per half hour)'] = pd.to_numeric(
    df['KWH/hh (per half hour)'], errors='coerce'
)

# 2️⃣  Sort by meter + time *and reset the row order*
df = (
    df.sort_values(['LCLid', 'DateTime'])     # put rows in time order
      .reset_index(drop=True)                 # now index 0-N matches that order
)

# 3️⃣  Forward-fill then back-fill within each meter
df['KWH/hh (per half hour)'] = (
    df.groupby('LCLid')['KWH/hh (per half hour)']
      .transform(lambda s: s.ffill().bfill())
)

df['KWH/hh (per half hour)'] = df['KWH/hh (per half hour)'].fillna(0)        # or .mean()

# 4️⃣  Confirm all gone
assert df['KWH/hh (per half hour)'].isna().sum() == 0

# 1.1 normalize
# Assuming df is the DataFrame from your provided code
# Normalize KWH/hh (per half hour) for each LCLid using Min-Max scaling
df['KWH/hh (per half hour)'] = df.groupby('LCLid')['KWH/hh (per half hour)'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
)

# Handle cases where max = min (to avoid division by zero)
# If max = min, the normalized value is set to 0 (or another constant, if preferred)
df['KWH/hh (per half hour)'] = df['KWH/hh (per half hour)'].fillna(0)


# ----------------------------------------------------------------------
# 2.  ONE 48-ELEMENT PROFILE PER ID-DAY   (**TWO FIXES HERE**)
# ----------------------------------------------------------------------
template = np.arange(48, dtype=np.uint8)        # slots 0 … 47

def build_profile(sub):
    # ---- FIX-1:  reindex on template so the array is ALWAYS length-48
    s = (sub.groupby("slot")["KWH/hh (per half hour)"]
           .mean()                              # collapse duplicates, if any
           .reindex(template, fill_value=0.0))  # fill missing half-hours
    return s.to_numpy(dtype=np.float32)

profiles = (df
            .groupby(["LCLid", "Date"])          # no include_groups kwarg
            .apply(build_profile)
            .to_frame("load_curve")
            .reset_index())

# ---- FIX-2:  (sanity) assert every curve really is length 48
assert profiles["load_curve"].map(len).eq(48).all(), "non-48 curve slipped in!"

# ----------------------------------------------------------------------
# 3.  CALENDAR FEATURES PER ID-DAY
# ----------------------------------------------------------------------
calendar = (df.drop_duplicates(subset=["LCLid","Date"])
              [["LCLid","Date","Month","DayOfWeek","DayOfMonth","WeekOfYear"]])
dataset  = profiles.merge(calendar, on=["LCLid","Date"], how="left")

# ----------------------------------------------------------------------
# 4.  ENCODE CONDITIONS  (pick ONE path)
# ----------------------------------------------------------------------
# ---- Path A: one-hot (for tree models) -------------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
X_cat = ohe.fit_transform(dataset[["LCLid","Month","DayOfWeek","DayOfMonth"]])
X     = sparse.hstack([X_cat, dataset[["WeekOfYear"]].to_numpy()])

# ---- Path B: cyclical + ID index (for neural nets) -------------------
dataset["Mo_sin"]  = np.sin(2*np.pi*dataset["Month"]/12)
dataset["Mo_cos"]  = np.cos(2*np.pi*dataset["Month"]/12)
dataset["DoW_sin"] = np.sin(2*np.pi*dataset["DayOfWeek"]/7)
dataset["DoW_cos"] = np.cos(2*np.pi*dataset["DayOfWeek"]/7)
dataset["DoM_sin"] = np.sin(2*np.pi*dataset["DayOfMonth"]/31)
dataset["DoM_cos"] = np.cos(2*np.pi*dataset["DayOfMonth"]/31)

X_dense   = dataset[["Mo_sin","Mo_cos","DoW_sin","DoW_cos",
                     "DoM_sin","DoM_cos","WeekOfYear"]].to_numpy()
LCLid_idx = dataset["LCLid"].astype("category").cat.codes.to_numpy()

# ----------------------------------------------------------------------
# 5.  TARGET  y  (N × 48)   – no longer fails
# ----------------------------------------------------------------------
y = np.stack(dataset["load_curve"].values, axis=0)    # guaranteed equal shapes

# ----------------------------------------------------------------------
# 6.  TRAIN / VALIDATION SPLIT   (uses the one-hot matrix here)
# ----------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

import torch
# X_tr = torch.from_numpy(X_tr).float()
# X_val = torch.from_numpy(X_val).float()
# WARNING: converts the whole sparse matrix to dense
X_tr_dense  = X_tr.toarray().astype("float32")
X_val_dense = X_val.toarray().astype("float32")

X_tr_t  = torch.from_numpy(X_tr_dense)
X_val_t = torch.from_numpy(X_val_dense)

y_tr = torch.from_numpy(y_tr).float()
y_val = torch.from_numpy(y_val).float()

# print(y_tr.max(), y_val.max())
# max_y = max(y_tr.max(), y_val.max())
# y_val = y_val / max_y
# y_tr = y_tr / max_y

save_dir = f"./data/london/{num_users}"
os.makedirs(save_dir, exist_ok=True)
torch.save(X_tr_t, f"{save_dir}/X_tr.pt")
torch.save(X_val_t, f"{save_dir}/X_val.pt")
torch.save(y_tr, f"{save_dir}/y_tr.pt")
torch.save(y_val, f"{save_dir}/y_val.pt")
print("done")