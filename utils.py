import pandas as pd
import os.path
import hashlib
from logbook import Logger

LOG = Logger("utils")


def archive(df, name, path=".", types=["feather"]):
    """
    Convenience function for writing dataframe to a given location for
    later use, only archiving if the file doesn't exist.
    """
    hl = hashlib.sha256(pd.utils.hash_pandas_object(df).values.tobytes())
    h = hl.hexdigest()[:8]
    bn = os.path.basename(name)
    bn = os.path.splitext(bn)[0]
    for t in types:
        if t == "feather":
            ofn = os.path.join(path, f"{bn}-{h}.feather")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.to_feather(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        elif t == "parquet":
            ofn = os.path.join(path, f"{bn}-{h}.parquet")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.to_parquet(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        elif t == "csv":
            ofn = os.path.join(path, f"{bn}-{h}.csv")
            if not os.path.exists(ofn):
                LOG.info(f'Archiving DataFrame to "{ofn}"')
                df.to_csv(ofn)
            else:
                LOG.info(f'Archived DataFrame "{ofn}" already exists: skipping')
        else:
            raise ValueError(f'Archive type "{t}" not understood')
