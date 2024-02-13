import pandas as pd


def data_filter(df):
    # # This function pops data from df where patients only have 1 scan (i.e. sum < 2)
    #sum rows excluding column 1 (patient ID)
    row_sums = df.iloc[:, 1:].sum(axis=1)
    # Get the indices of rows where the sum is less than 2
    rows_to_drop = row_sums[row_sums < 2].index
    # Drop rows from DataFrame
    df = df.drop(index=rows_to_drop)

    return df


