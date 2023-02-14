import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from data import Data
import utils
from encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split


class Trainer:

    def __init__(self, data):
        self.data = data
        self.pipeline = self.set_pipeline()
        # Hold out ( train and test dplit )
        X = data.drop('fare_amount', axis=1)
        y = data['fare_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.run()
        self.y_pred = self.pipeline.predict(self.X_test)


    def set_pipeline(self):
        '''returns a pipelined model'''
        # A  COMPLETER
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance',
                                           dist_pipe,
                                           ["pickup_latitude",
                                            "pickup_longitude",
                                            'dropoff_latitude',
                                            'dropoff_longitude']),
                                          ('time',
                                           time_pipe,
                                           ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        return pipe



    def run(self):
        self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self):
        rmse = np.sqrt(np.mean((self.y_test - self.y_pred) ** 2))
        return rmse


data = Data("data/train.csv", 10000)
data.clean_data()

trainer = Trainer(data.get_data())

print(trainer.evaluate())
