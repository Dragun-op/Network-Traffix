"""
row_parser.py -- turn a pasted raw CICIDS-2017 CSV line into the flow dict the
scorer consumes.

A judge pastes any row copied from a CICIDS CSV. The values are in the fixed
column order below, so we zip them back onto the column names. The row may or
may not include the trailing Label; if it does, we return it as `ground_truth`
so the UI can show "predicted vs actual" (a nice touch, not used by the model).

Feature selection / imputation is NOT done here -- the flow dict is handed to the
same FeatureExtractor the model trained with, so a pasted row goes through the
identical transform as everything else.
"""
from __future__ import annotations

import math

# Exact CICIDS-2017 column order (as pandas reads the header; the duplicate
# 'Fwd Header Length' becomes 'Fwd Header Length.1'). Position -> feature name.
CICIDS_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
    "Idle Min", "Label",
]
FEATURE_COLUMNS = CICIDS_COLUMNS[:-1]   # everything except the trailing Label
N_WITH_LABEL = len(CICIDS_COLUMNS)      # 79
N_NO_LABEL = len(FEATURE_COLUMNS)       # 78


def _to_number(x: str):
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return math.nan
    try:
        return float(x)          # handles "Infinity", "1.0", "38308", etc.
    except ValueError:
        return x                 # leave non-numeric as-is; transform will coerce


def parse_row(row: str) -> tuple[dict, str | None]:
    """Parse a pasted CICIDS row. Returns (flow_dict, ground_truth_label_or_None).
    Raises ValueError with a helpful message if the column count is wrong."""
    parts = [p for p in row.strip().split(",")]
    n = len(parts)

    if n == N_WITH_LABEL:
        ground_truth = parts[-1].strip() or None
        values = parts[:-1]
        names = FEATURE_COLUMNS
    elif n == N_NO_LABEL:
        ground_truth = None
        values = parts
        names = FEATURE_COLUMNS
    else:
        raise ValueError(
            f"Expected {N_NO_LABEL} feature columns (or {N_WITH_LABEL} with a Label), "
            f"but got {n}. Paste a full CICIDS-2017 flow row."
        )

    flow = {name: _to_number(v) for name, v in zip(names, values)}
    return flow, ground_truth