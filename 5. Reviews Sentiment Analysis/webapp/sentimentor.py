from datetime import datetime

from joblib import load
from pandas import DataFrame

"""
-----------------------------------
------ 2. SENTIMENTOR CLASS -------
-----------------------------------
"""


class Sentimentor:

    def __init__(self, input_data, product_id=None):
        self.input_data = input_data
        self.product_id = product_id
        self.pipeline = load('pipelines/text_prep_pipeline.pkl')
        self.model = load('models/sentiment_clf_model.pkl')

    def prep_input(self):
        # Verify if the type of input data
        if isinstance(self.input_data, str):
            self.input_data = [self.input_data]
        elif isinstance(self.input_data, DataFrame):
            self.input_data = list(self.input_data.iloc[:, 0].values)

        # Apply the pipeline to prepare the input data
        return self.pipeline.transform(self.input_data)

    def make_predictions(self):
        # Preparing the data and calling the classifier for making predictions
        text_list = self.prep_input()
        pred = self.model.predict(text_list)
        proba = self.model.predict_proba(text_list)[:, 1]

        # Analyzing the results and preparing the output
        class_sentiment = ['Positive' if c == 1 else 'Negative' for c in pred]
        class_proba = [p if c == 1 else 1 - p for c, p in zip(pred, proba)]

        # Building up a pandas DataFrame to deliver the results
        results = {
            'datetime_prediction': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'product_id': self.product_id,
            'text_input': self.input_data,
            'prediction': pred,
            'class_sentiment': class_sentiment,
            'class_probability': class_proba
        }

        df_results = DataFrame(results)

        # Exporting results
        df_results.to_csv('sentiment_predictions.csv', index=False)

        return df_results
