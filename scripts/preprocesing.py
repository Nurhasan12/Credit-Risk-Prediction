import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_dataset(path):
    return pd.read_csv(path)

def make_target(df):
    df['target'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
    return df

def drop_useless_columns(df):
    drop_cols = ['id', 'member_id', 'url', 'desc', 'loan_status']
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def encode_and_scale(df):
    df = df.copy()

    # Tangani kolom kategorikal
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        df[col] = LabelEncoder().fit_transform(df[col])

    # Tangani kolom numerik
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != 'target']

    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Pastikan tidak ada NaN
    df = df.dropna()

    return df


