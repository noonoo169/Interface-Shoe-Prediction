import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model = pickle.load(open("models/sneaker-prediction_random-forest.pkl", 'rb'))
category_encoder = pickle.load(open("models/Category_encoder.pkl", 'rb'))
branch_encoder = pickle.load(open("models/Branch_encoder.pkl", 'rb'))

def get_encoder_class():
    return category_encoder.classes_ , branch_encoder.classes_

def model_predict(input_data):
    try:
        new_input = input_data.copy()
        new_input['category'] = category_encoder.transform([input_data['category']])
        new_input['branch'] = branch_encoder.transform([input_data['branch']])
        input_data_df = pd.DataFrame([new_input])
        prediction = model.predict(input_data_df)
    except Exception as e:
        return 'Can not predict your data'
    else:
        return prediction

    