import pandas as pd
from joblib import load

from log.log_config import *
from sentimentor import Sentimentor

"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Creating a logging object
logger = logging.getLogger(__name__)
logger = logger_config(logger, level=logging.DEBUG, filemode='a')

# Messages
WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'

"""
-----------------------------------
-------- 3. MAIN PROGRAM ----------
  3.1 Making Predictions on Text
-----------------------------------
"""

if __name__ == '__main__':

    # Training the model (if applicable)
    if bool(True):
        logger.info('Starting train.py script')

    # Reading pkl files
    logger.debug('Reading pkl files')
    try:
        pipeline = load('./pipelines/text_prep_pipeline.pkl')
        model = load('./models/sentiment_clf_model.pkl')
    except Exception as e:
        logger.error(e)
        logger.warning(WARNING_MESSAGE)
        exit()

    # Fake input
    text_input_df = pd.read_csv('../../Data Source/olist_master.csv')
    # Convert review_comment_message column to strings
    text_input_df['review_comment_message'] = text_input_df['review_comment_message'].astype(str)

    # Extract product_id column
    product_ids = text_input_df['product_id'].tolist()

    # Extract review_comment_message column
    text_input = text_input_df['review_comment_message'].tolist()

    # Instancing an object and executing predictions
    logger.debug('Creating a sentimentor object and making predictions')
    sentimentor = Sentimentor(input_data=text_input, product_id=product_ids)
    try:
        output = sentimentor.make_predictions()
        logger.info('Module finished with success status')
        exit()
    except Exception as e:
        logger.error(e)
        logger.warning(WARNING_MESSAGE)
        exit()
