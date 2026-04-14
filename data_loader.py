import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
from config import CFG

def load_data(path: str) -> Tuple[List[str], np.ndarray, LabelEncoder]:
    df     = pd.read_csv(path)
    texts  = df[CFG.TEXT_COL].astype(str).tolist()
    labels = df[CFG.LABEL_COL].astype(str).tolist()
    le     = LabelEncoder()
    y      = le.fit_transform(labels)
    print(f"\nDataset loaded: {len(texts)} samples")
    print(f"Classes: {list(le.classes_)}")
    print(pd.Series(labels).value_counts().to_string())
    return texts, y, le