import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import argparse
import joblib
import warnings
import gc

warnings.filterwarnings("ignore")

CHUNK = 1000  
LABEL = "label"

def log(x):
    print(f"[LOG] {x}")

# ============================================================
# PASS 1 – SCAN DATASET
# ============================================================
def scan_dataset(input_csv):
    log("=== PASS 1: scanning dataset ===")

    scaler = StandardScaler()
    unique_labels = set()
    
    vocab = {}           
    numeric_cols = []    
    cat_cols = []        
    
    first_chunk = True
    chunk_id = 0

    reader = pd.read_csv(input_csv, chunksize=CHUNK)

    for chunk in reader:
        chunk_id += 1
        log(f"Scanning chunk {chunk_id}")

        chunk.columns = [c.lower() for c in chunk.columns]
        
        # 1. Thu thập Label
        if LABEL in chunk.columns:
            unique_labels.update(chunk[LABEL].astype(str).unique())
        else:
            raise ValueError(f"Dataset missing label column: '{LABEL}'")

        # 2. Xác định cột (Chỉ chunk đầu)
        if first_chunk:
            cat_cols = chunk.select_dtypes(include=["object", "category"]).columns.tolist()
            if LABEL in cat_cols: cat_cols.remove(LABEL)
            
            numeric_cols = chunk.select_dtypes(exclude=["object", "category"]).columns.tolist()
            if LABEL in numeric_cols: numeric_cols.remove(LABEL)
            
            for col in cat_cols:
                vocab[col] = set()
            first_chunk = False

        # 3. Update Vocab
        for col in cat_cols:
            vocab[col].update(chunk[col].dropna().astype(str).unique())

        # 4. Partial Fit Scaler
        # --- FIX LỖI INFINITY TẠI ĐÂY ---
        # B1: Coerce sang số (chuỗi lỗi thành NaN)
        num_data = chunk[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # B2: Thay thế Infinity/-Infinity bằng NaN trước, hoặc bằng 0 luôn
        # Scikit-learn không chấp nhận Inf, nên ta đưa về 0 (hoặc giá trị cực đại tùy ý, nhưng 0 an toàn nhất cho scale)
        num_data.replace([np.inf, -np.inf], 0, inplace=True)
        
        # B3: Fill NaN bằng 0
        num_data.fillna(0, inplace=True)
        
        if len(num_data) > 0:
            scaler.partial_fit(num_data)
        
        # Dọn rác
        del chunk
        del num_data
        gc.collect()

    # Sắp xếp vocab
    for col in vocab:
        vocab[col] = sorted(list(vocab[col]))

    log("Fitting label encoder...")
    le = LabelEncoder()
    le.fit(sorted(list(unique_labels)))

    return scaler, le, vocab, numeric_cols, cat_cols


# ============================================================
# PASS 2 – TRANSFORM DATASET
# ============================================================
def process_dataset(input_csv, output_csv, scaler, le, vocab, numeric_cols, cat_cols):
    log("=== PASS 2: processing dataset ===")
    
    first_write = True
    chunk_id = 0

    reader = pd.read_csv(input_csv, chunksize=CHUNK)

    for chunk in reader:
        chunk_id += 1
        log(f"Processing chunk {chunk_id}")

        chunk.columns = [c.lower() for c in chunk.columns]

        # 1. Process Label
        y = le.transform(chunk[LABEL].astype(str))

        # 2. Process Numeric
        # --- FIX LỖI INFINITY TẠI ĐÂY (Lặp lại logic Pass 1) ---
        X_num = chunk[numeric_cols].apply(pd.to_numeric, errors='coerce')
        X_num.replace([np.inf, -np.inf], 0, inplace=True) # Fix Infinity
        X_num.fillna(0, inplace=True)                     # Fix NaN
        
        # Scale
        X_num_scaled = scaler.transform(X_num)
        X_num_df = pd.DataFrame(X_num_scaled, columns=numeric_cols, index=chunk.index)

        del X_num
        gc.collect()

        # 3. Process Categorical (One-Hot)
        cat_dfs = []
        for col in cat_cols:
            cat_series = pd.Categorical(
                chunk[col].astype(str), 
                categories=vocab[col], 
                ordered=True
            )
            dummies = pd.get_dummies(cat_series, prefix=col) 
            cat_dfs.append(dummies)
        
        if cat_dfs:
            X_cat_df = pd.concat(cat_dfs, axis=1)
        else:
            X_cat_df = pd.DataFrame(index=chunk.index)

        del chunk
        gc.collect()

        # 4. Gộp
        X_num_df.reset_index(drop=True, inplace=True)
        X_cat_df.reset_index(drop=True, inplace=True)
        
        final_chunk = pd.concat([X_num_df, X_cat_df], axis=1)
        final_chunk[LABEL] = y

        del X_num_df
        del X_cat_df
        gc.collect()

        # 5. Ghi file
        final_chunk.to_csv(
            output_csv,
            mode="a",
            index=False,
            header=first_write
        )
        first_write = False
        
        del final_chunk
        gc.collect()

    log(f"=== DONE: saved to {output_csv} ===")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="processed.csv")
    parser.add_argument("--artefacts", default="artefacts")

    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)
    artefact_dir = Path(args.artefacts)
    artefact_dir.mkdir(exist_ok=True, parents=True)

    # PASS 1
    scaler, le, vocab, num_cols, cat_cols = scan_dataset(input_csv)

    joblib.dump(scaler, artefact_dir / "scaler.pkl")
    joblib.dump(le, artefact_dir / "label_encoder.pkl")
    joblib.dump(vocab, artefact_dir / "vocab.pkl")
    joblib.dump(num_cols, artefact_dir / "numeric_cols.pkl")
    joblib.dump(cat_cols, artefact_dir / "cat_cols.pkl")

    # PASS 2
    process_dataset(input_csv, output_csv, scaler, le, vocab, num_cols, cat_cols)