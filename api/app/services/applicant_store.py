"""Read-only access to the model-ready applicant tables (Phase 5 output).

The tables are large (307,511 x 258 and 48,744 x 257), so rows are read with
pyarrow predicate push-down instead of loading the whole file into memory.
"""

from functools import lru_cache
from typing import List, Optional

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from api.app.core.config import MODEL_INPUT_PATH, MODEL_INPUT_TEST_PATH, SPLITS_DIR

ID_COL = "SK_ID_CURR"


def _ids_from_parquet(path, column: str = ID_COL) -> List[int]:
    if not path.exists():
        return []
    return pq.read_table(path, columns=[column]).column(column).to_pylist()


@lru_cache(maxsize=1)
def labelled_ids() -> frozenset:
    """All applicant IDs in the labelled model-input table (application_train population)."""
    return frozenset(_ids_from_parquet(MODEL_INPUT_PATH))


@lru_cache(maxsize=1)
def inference_ids() -> frozenset:
    """All applicant IDs in the unlabelled inference table (application_test population)."""
    return frozenset(_ids_from_parquet(MODEL_INPUT_TEST_PATH))


@lru_cache(maxsize=1)
def test_split_ids() -> List[int]:
    """Frozen test-partition IDs (Phase 6 split, seed 42) in stored order."""
    return _ids_from_parquet(SPLITS_DIR / "test_indices.parquet")


def partition_of(applicant_id: int) -> Optional[str]:
    """Return 'labelled' or 'inference' depending on which table holds the applicant, else None."""
    if applicant_id in labelled_ids():
        return "labelled"
    if applicant_id in inference_ids():
        return "inference"
    return None


def load_applicants(ids: List[int]) -> pd.DataFrame:
    """Load full feature rows for the given applicant IDs from whichever table holds them."""
    frames = []
    for path in (MODEL_INPUT_PATH, MODEL_INPUT_TEST_PATH):
        if not path.exists():
            continue
        table = pq.read_table(path, filters=[(ID_COL, "in", list(ids))])
        if table.num_rows:
            frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # preserve the requested order
    order = {aid: i for i, aid in enumerate(ids)}
    df["_order"] = df[ID_COL].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def load_columns(columns: List[str], path=MODEL_INPUT_PATH) -> pd.DataFrame:
    """Load a few columns of a whole table (used for dataset-level statistics)."""
    return pq.read_table(path, columns=columns).to_pandas()


def search_ids(query: str, limit: int = 8) -> List[dict]:
    """Prefix search over applicant IDs (labelled and inference tables)."""
    q = query.strip().upper().replace("APP-", "")
    if not q.isdigit():
        return []
    hits = []
    for pool, source in ((labelled_ids(), "labelled"), (inference_ids(), "inference")):
        for aid in pool:
            if str(aid).startswith(q):
                hits.append({"sk_id_curr": aid, "applicant_id": f"APP-{aid}", "source": source})
                if len(hits) >= limit:
                    return sorted(hits, key=lambda h: h["sk_id_curr"])
    return sorted(hits, key=lambda h: h["sk_id_curr"])
