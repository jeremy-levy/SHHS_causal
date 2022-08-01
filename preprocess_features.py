from sklearn.feature_selection import VarianceThreshold


def remove_null_features(data):
    list_features_name = data.columns.values
    features_null = list_features_name[data.isna().sum() > data.shape[0] / 4]

    data = data.drop(columns=features_null)
    return data


def variance_threshold(data, threshold=0.2):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def preprocess_data(data):
    data = remove_null_features(data)
    data = variance_threshold(data)

    return data
