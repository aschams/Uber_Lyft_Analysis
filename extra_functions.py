def get_avg_by_cab(df: pd.DataFrame,
                   cab_type: str,
                   gb_feature: str,
                   feature: str) -> pd.DataFrame:
    """
    Returns a dataframe for the average value of a given feature based on a groupby feature.
    Inputs:
        df: Source DataFrame
        cab_type: Type of cab; 'Lyft' or 'Uber'
        gb_feature: feature to groupby df
        feature: feature to take the mean of
    Outputs:
        avg_price: df of the average value of feature, grouped by gb_feature
    """
    assert cab_type == 'Lyft' or cab_type == 'Uber','Invalid cab_type specified'
    avg_by_timestamp = df[df.cab_type == cab_type].groupby(by=gb_feature).mean()
    avg = avg_by_timestamp[feature]
    avg.index = pd.to_datetime(avg.index)
    avg_price = avg.groupby(pd.Grouper(freq='h')).ffill()
    return avg_price
