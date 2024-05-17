import re
from matplotlib import pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin


def re_breakline(text_list, text_sub=' '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    return [re.sub('[\n\r]', text_sub, r) for r in text_list]


def re_hiperlinks(text_list, text_sub=' link '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_dates(text_list, text_sub=' date '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_money(text_list, text_sub=' money '):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_numbers(text_list, text_sub=' numbers '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub('[0-9]+', text_sub, r) for r in text_list]


def re_negation(text_list, text_sub=' negation '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub("(?i)\b(?:n't|never|no|nothing|nowhere|none|not)\b", text_sub, r) for r in text_list]


def re_special_chars(text_list, text_sub=' '):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    text_sub: string or pattern to substitute the regex pattern [type: string]
    """

    # Applying regex
    return [re.sub('\W', text_sub, r) for r in text_list]


def re_whitespaces(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """

    # Applying regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end


def stemming_process(text, stemmer=SnowballStemmer('english')):
    """
    Args:
    ----------
    text: list object where the stopwords will be removed [type: list]
    stemmer: type of stemmer to be applied [type: class, default: RSLPStemmer()]
    """

    return [stemmer.stem(c) for c in text.split()]


class ApplyRegex(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for applying multiple regex transformations to text data.

    Args:
        regex_transformers (dict): A dictionary where keys are names for the transformations
            and values are functions that accept a list of strings (text data) and return
            a modified list of strings with the regex applied.

    Attributes:
        regex_transformers (dict): The dictionary of regex transformations provided during initialization.
    """

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        """Fits the transformer to the data. Does nothing in this case."""
        return self

    def transform(self, X, y=None):
        """Applies the defined regex transformations to the text data in X."""
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)
        return X


class StemmingProcess(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for applying stemming to text data.

    Args:
        stemmer: A stemming function or object that takes a single word as input
            and returns its stemmed form.

    Attributes:
        stemmer: The stemming function or object provided during initialization.
    """

    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        """Fits the transformer to the data. Does nothing in this case."""
        return self

    def transform(self, X, y=None):
        """
        Applies stemming to each word in the text data and returns the stemmed text.

        Args:
            X (list of str): The list of text documents to be stemmed.

        Returns:
            list of str: The list of stemmed text documents.
        """
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]


class TextFeatureExtraction(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for extracting features from text data using a vectorizer.

    Args:
        vectorizer: A scikit-learn vectorizer object (e.g., CountVectorizer, TfidfVectorizer)
            used to transform text into numerical features.
        train (bool, optional): Indicates whether the transformer is being used in training mode
            (fit and transform) or inference mode (transform only). Defaults to True (training mode).

    Attributes:
        vectorizer: The vectorizer object provided during initialization.
        train (bool): Whether the transformer is in training or inference mode.
    """

    def __init__(self, vectorizer, train=True):
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, X, y=None):
        """
        Fits the vectorizer to the training data X if in training mode.

        Args:
            X (list of str): The list of text documents to fit the vectorizer on.
            y (None): Not used in this context.

        Returns:
            self: Returns the fitted transformer object.
        """
        if self.train:
            self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        """
        Transforms the text data X into numerical features using the fitted vectorizer.

        Args:
            X (list of str): The list of text documents to transform.
            y (None): Not used in this context.

        Returns:
            sparse matrix or array: The transformed feature matrix.
        """
        if self.train:
            return self.vectorizer.fit_transform(X).toarray()
        else:
            return self.vectorizer.transform(X)


# Defining a function to plot the sentiment of a given phrase
def sentiment_analysis(text, pipeline, vectorizer, model):
    """
    Performs sentiment analysis on a given text using a trained pipeline and model.

    Args:
        text (str or list): The text string or list of strings to analyze.
        pipeline (sklearn.Pipeline): The text preprocessing pipeline to apply.
        vectorizer: The vectorizer used to transform text into features.
        model: The trained classification model for sentiment prediction.

    Returns:
        None: Displays a plot visualizing the sentiment and its probability score.
    """

    # Ensure text is a list for consistent processing
    if isinstance(text, str):
        text = [text]

    # Apply preprocessing pipeline and vectorization
    text_prep = pipeline.transform(text)  # No need to fit again
    matrix = vectorizer.transform(text_prep)

    # Predict sentiment and probability
    pred = model.predict(matrix)[0]  # Get prediction for the single input
    proba = model.predict_proba(matrix)[0]  # Get probabilities

    # Visualize sentiment with probability score
    fig, ax = plt.subplots(figsize=(5, 3))
    text_color = 'seagreen' if pred == 1 else 'crimson'  # Color based on sentiment
    sentiment_text = 'Positive' if pred == 1 else 'Negative'
    class_proba = 100 * round(proba[pred], 2)  # Probability of predicted class

    ax.text(0.5, 0.5, sentiment_text, fontsize=50, ha='center', color=text_color)
    ax.text(0.5, 0.20, f"{class_proba}%", fontsize=14, ha='center')
    ax.axis('off')
    ax.set_title('Sentiment Analysis', fontsize=14)
    plt.show()
