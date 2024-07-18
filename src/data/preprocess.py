import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def fill_with_median(group):
    group.fillna(group.median(), inplace=True)
    return group

def preprocess(train, val):
    columns_to_alter = [col for col in train.columns if col != "class"]

    train.replace("na", np.nan, inplace=True)

    train[columns_to_alter] = train[columns_to_alter].astype("float64")

    val.replace("na", np.nan, inplace=True)

    val[columns_to_alter] = val[columns_to_alter].astype("float64")
    
    features = columns_to_alter
    target = ['class']

    X = train[features]
    y = train[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train_completed = X_train
    X_train_completed[y_train['class'] == 'neg'] = X_train[y_train['class'] == 'neg'].apply(fill_with_median)
    X_train_completed[y_train['class'] == 'pos'] = X_train[y_train['class'] == 'pos'].apply(fill_with_median)

    X_test_completed = X_test
    X_test_completed[y_test['class'] == 'neg'] = X_test[y_test['class'] == 'neg'].apply(fill_with_median)
    X_test_completed[y_test['class'] == 'pos'] = X_test[y_test['class'] == 'pos'].apply(fill_with_median)

    X_val = val[features]
    y_val = val[target]

    X_val = X_val.loc[:]
    y_val = y_val.loc[:]

    X_val_completed = X_val
    X_val_completed[y_val['class'] == 'neg'] = X_val.loc[y_val['class'] == 'neg'].apply(fill_with_median)
    X_val_completed[y_val['class'] == 'pos'] = X_val.loc[y_val['class'] == 'pos'].apply(fill_with_median)

    scaler = StandardScaler()

    X_train_normalized = scaler.fit_transform(X_train_completed)
    X_test_normalized = scaler.transform(X_test_completed)
    X_val_normalized = scaler.transform(X_val_completed)

    rus = RandomUnderSampler(random_state=42)
    X_train_res_under, y_train_res_under = rus.fit_resample(X_train_normalized, y_train)

    print("--------------------------------------------------------------------------------")
    print("Preprocessing completed.")
    print("Train shape after under-sampling:", X_train_res_under.shape)
    print("Test shape:", X_test_normalized.shape)
    print("Validation shape:", X_val_completed.shape)
    print("--------------------------------------------------------------------------------")
    
    return X_train_res_under, X_test_normalized, y_train_res_under, y_test, X_val_normalized, y_val
