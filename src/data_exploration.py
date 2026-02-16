import pandas as pd
from collections import OrderedDict

# -------------------- DATA EXPLORATION --------------------
def data_exploration(df):

    numerical_col = df.select_dtypes(exclude='object').columns
    categorical_col = df.select_dtypes(include = 'object').columns

    numerical_stats = []
    categorical_stats = []

   # ---------- Numerical Analysis ----------
    Q1 = df[numerical_col].quantile(0.25)
    Q3 = df[numerical_col].quantile(0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5 * IQR
    UW = Q3 + 1.5 * IQR
    outlier_count = ((df[numerical_col] < LW) | (df[numerical_col] > UW))


    for col in numerical_col:
        num_stats = ({
            "feature": col,
            "mean": df[col].mean(),
            "median": df[col].median(),
            "quartile_1": Q1[col],
            "quartile_3": Q3[col],
            "IQR": IQR[col],
            "lower_whisker": LW[col],
            "upper_whisker": UW[col],
            "outlier_count": outlier_count[col].sum(),
            "std_dev": df[col].std(),
            "variance": df[col].var(),
            "skewness": df[col].skew(),
            "kurtosis": df[col].kurtosis()
        })

        numerical_stats.append(num_stats)

    # ---------- Categorical Analysis ----------

    for col in categorical_col:
        cat_stats = ({
            "feature": col,
            "unique_values": df[col].nunique(),
            "mode": df[col].mode()[0] if not df[col].mode().empty else None,
            "missing_values": df[col].isnull().sum()
        })

        categorical_stats.append(cat_stats)

    # return both reports
    numerical_report = pd.DataFrame(numerical_stats)
    categorical_report = pd.DataFrame(categorical_stats)

    return numerical_report, categorical_col    
        

    