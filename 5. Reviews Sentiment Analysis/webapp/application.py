import pandas as pd
from googletrans import Translator
from log.log_config import *


def translate_reviews(predictions):
    translator = Translator()
    translations = []

    for review in predictions['review']:
        try:
            translation = translator.translate(review, src='pt', dest='en')
            translated_text = translation.text
        except Exception as e:
            logger.error(f"Error translating review: {e}")
            translated_text = ""  # Set translated text to empty string in case of error
        translations.append(translated_text)

    predictions['review'] = translations
    return predictions


def main():
    # Load the sentiment predictions CSV file
    predictions = pd.read_csv("sentiment_predictions.csv")
    predictions = predictions.dropna(subset=['review'])

    # Translate reviews from Portuguese to English
    predictions = translate_reviews(predictions)

    # Save the translated reviews to the CSV file
    predictions.to_csv("sentiment_predictions.csv", index=False)


if __name__ == "__main__":
    # Creating a logging object
    logger = logging.getLogger(__name__)
    logger = logger_config(logger, level=logging.DEBUG, filemode='a')

    # Messages
    WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'

    # Starting the translation process
    try:
        logger.info('Translating reviews...')
        main()
        logger.info('Translation completed successfully.')
    except Exception as e:
        logger.error(e)
        logger.warning(WARNING_MESSAGE)






# import streamlit as st
# import pandas as pd
# from log.log_config import *
#
#
# # Load the sentiment predictions CSV file
# def load_data():
#     predictions = pd.read_csv("sentiment_predictions.csv")
#     predictions = predictions.dropna(subset=['review'])
#     # Rename columns
#     # predictions.rename(columns={'text_input': 'review', 'class_sentiment': 'sentiment'}, inplace=True)
#     return predictions
#
#
# # Display the sentiment predictions DataFrame
# def display_predictions(predictions):
#     st.write("Sentiment Predictions:")
#     st.dataframe(predictions, hide_index=True)
#
#
# # Filter reviews by product ID
# def filter_reviews(predictions, product_id):
#     filtered_reviews = predictions[predictions['product_id'] == product_id]
#     return filtered_reviews[['review', 'sentiment']]
#
#
# def main():
#     # Load data
#     predictions = load_data()
#
#     # Display sentiment predictions
#     display_predictions(predictions)
#
#     # Allow users to search for reviews by product ID
#     st.sidebar.title("Search Reviews by Product ID")
#     product_id = st.sidebar.text_input("Enter Product ID:")
#     if product_id:
#         filtered_reviews = filter_reviews(predictions, product_id)
#         if not filtered_reviews.empty:
#             st.subheader(f"Reviews for Product ID: {product_id}")
#             st.write(filtered_reviews)
#         else:
#             st.write("No reviews found for the entered Product ID.")
#
#
# if __name__ == "__main__":
#     # Creating a logging object
#     logger = logging.getLogger(__name__)
#     logger = logger_config(logger, level=logging.DEBUG, filemode='a')
#
#     # Messages
#     WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'
#
#     # Starting the Streamlit app
#     try:
#         logger.info('Starting Streamlit app...')
#         main()
#     except Exception as e:
#         logger.error(e)
#         logger.warning(WARNING_MESSAGE)
