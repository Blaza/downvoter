from preprocessing import process_body, process_title
import pandas as pd

from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores = cpu_count()


def process_csv(csv_file, out_file=None):
    print("Loading csv data")
    data = pd.read_csv(csv_file)
    print("Making apply operation")
    data = dd.from_pandas(data, npartitions=nCores)
    data["body"] = data.map_partitions(\
            lambda df: df.apply(\
                lambda x: process_body(x.body), axis=1))
    data["title"] = data.map_partitions(\
            lambda df: df.apply(\
                lambda x: process_title(x.title), axis=1))
    print("processing data")
    output = data.compute(get=get)
    # write to out_file if provided
    print("Exporting to csv")
    if out_file is not None:
        output.dropna(how="any", inplace=True)
        output.to_csv(out_file)

    print("Finished")
    return output

if __name__ == "__main__":
    print("Running csv processing")
    prefixes = ["wv_train", "wv_val", "lr_train", "lr_val", "lr_test"]
    data_root = "/data/SO_data/downvoter"

    for prefix in prefixes:
        print("Processing %s" % prefix)
        in_name = "_".join([prefix, "raw_data.csv"])
        out_name = "_".join([prefix, "processed_data.csv"])
        process_csv("/".join([data_root, in_name]),
                    "/".join([data_root, out_name]))

    print("ALL DONE!")

