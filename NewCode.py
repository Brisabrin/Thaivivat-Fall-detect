from pyts.multivariate.transformation import WEASELMUSE
from tslearn.utils import to_pyts_dataset
# from sklearn.linear_model import LogisticRegression
from pickle5 import pickle
# unit time
sensor_data = [[[Xacc, Zacc, XGy, Zgy], [Xacc, Zacc, XGy, Zgy],
                [Xacc, Zacc, XGy, Zgy], [Xacc, Zacc, XGy, Zgy]]]


# min length 2400 laew koi fit model
if len(sensor_data[0]) >= 2400:
    # run code below

elif len(sensor_data[0]) > 36000:
    # delete beginning
    # then append new sensor data

for i in range(36000 - len(sensor_data[0])):

    sensor_data[0].append([0, 0, 0, 0])

sensor_data = to_pyts_dataset(sensor_data)

transformer = pickle.load(open('WEASELM-TRANSFORMER.sav', 'rb'))

sensor_data = transformer.transform(sensor_data)

model = pickle.load(open('WEASEL-MUSE.sav', 'rb'))
y_pred = model.predict(sensor_data)

if 'F' in y_pred:

    # trigger
    print()
