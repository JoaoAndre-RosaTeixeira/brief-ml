import pandas as pd


class Data:


    def __init__(self, file, nrows):
        self.data = self.init_data(file, nrows)

    def init_data(self, file, nrows):
        # import the train dataset ( 1000 first rows)
        df = pd.read_csv('data/train.csv', nrows=nrows)
        print(df.head())
        return df

    def get_data(self):
        return self.data

    # implement clean_data() function
    def clean_data(self, test=False):
        '''returns a DataFrame without outliers and missing values'''
        df = self.data
        df = df.dropna(how='any', axis='rows')
        df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
        df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
        if "fare_amount" in list(df):
            df = df[df.fare_amount.between(0, 4000)]
        df = df[df.passenger_count < 8]
        df = df[df.passenger_count >= 0]
        df = df[df["pickup_latitude"].between(left=40, right=42)]
        df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
        df = df[df["dropoff_latitude"].between(left=40, right=42)]
        df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
        self.data = df

