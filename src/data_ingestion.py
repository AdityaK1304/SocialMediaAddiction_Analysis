import pandas as pd
# data ingestion

def data_ingestion():
    df = pd.read_csv(r'C:\SocialMediaAddiction_Analysis\data\raw\Social_Media_Addiction.csv')
    return df