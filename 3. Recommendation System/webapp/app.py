import functools
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


class RecommendationSystem:
    def __init__(self):
        # Load data and models
        self.olist_db_ = pd.read_parquet('../../Data Source/olist_db_als.parquet')
        self.olist = pd.read_parquet('../../Data Source/olist_recommendation_dataset.parquet')
        self.olist_details_all = pd.read_parquet('../../Data Source/recomm_customer2vec.parquet')
        self.x = pd.read_parquet('../../Data Source/x_als.parquet')
        self.orders_details_control = pd.read_parquet('../../Data Source/loyalists.parquet')

        # Use pickle to load in the pre-trained model.
        with open('./model/SVD_ALS_model.pkl', 'rb') as f:
            SVD = pickle.load(f)

        decomposed_matrix = SVD.fit_transform(self.x)
        self.correlation_matrix = np.corrcoef(decomposed_matrix)

        # Define customer segments
        self.all_customer = list(self.olist_db_['Customer ID'].unique())
        self.olist_customers = list(self.olist['Customer ID'].unique())
        self.big_spenders = list(
            self.olist[self.olist['Customer Segment'] == 'Champions Big Spenders']['Customer ID'].unique())[:30]
        self.loyalists = list(self.orders_details_control['Customer ID'].unique())
        self.olist_all = list(self.olist_details_all['Customer ID'].unique())

    def recommendations_als(self, customer_id):
        """Generates product recommendations using ALS for a given customer."""
        fave_prod = self.olist_db_.groupby(['Customer ID']).max()['Product ID'].to_frame().reset_index()
        prd_id = fave_prod[fave_prod['Customer ID'] == customer_id]['Product ID']

        product_id = prd_id.iloc[0]
        product_names = list(self.x.index)
        product_id_idx = product_names.index(product_id)

        correlation_product_id = self.correlation_matrix[product_id_idx]
        recommend = list(self.x.index[correlation_product_id > 0.70])
        recommend.remove(product_id)

        predictions = pd.DataFrame(recommend[:20])
        predictions.columns = ['Product ID']

        predictions['Product Name'] = predictions['Customer ID'].apply(
            lambda x: self.olist_db_[self.olist_db_['Customer ID'] == x]['Product Name'].unique()[0])
        predictions['Average Rating'] = predictions['Customer ID'].apply(
            lambda x: self.olist_db_[self.olist_db_['Customer ID'] == x]['Ratings'].mean())

        recommendations = predictions[['Product Name', 'Average Rating']][:20]

        return recommendations

    def get_popular_products(self):
        """Gets the most popular products for the current month."""
        self.olist['Purchase Timestamp'] = pd.to_datetime(self.olist['Purchase Timestamp'], format='%Y-%m-%d %H:%M:%S')

        currentMonth = datetime.now().month

        self.olist['rating_month'] = self.olist['Purchase Timestamp'].apply(lambda x: x.month)
        temp = self.olist[self.olist.rating_month == currentMonth]
        popular_products = pd.DataFrame(
            temp.groupby(['rating_month', 'Product Name'], as_index=False).agg({'Review Score': ['count', 'mean']}))
        popular_products.columns = ['Rating Month', 'Product Name', 'Popularity', 'Average Review Ratings']
        popular_products = popular_products.sort_values('Popularity', ascending=False)

        return popular_products[['Product Name', 'Average Review Ratings']][:20]

    def upsell_products(self):
        """Recommends higher-priced products with good ratings for big spenders."""
        olist_Sp = self.olist[self.olist['Customer Segment'] == 'Champions Big Spenders']

        olist_Spenders = olist_Sp.groupby('Product Name', as_index=False).agg(
            {'Review Score': ['mean'], 'Monetary': 'mean'})

        olist_Spenders.columns = ['Product Name', 'Ratings', 'Price Value']
        olist_Spenders = olist_Spenders.sort_values(['Ratings', 'Price Value'], ascending=False)
        olist_Spenders.rename(columns={'Price Value': 'Price'}, inplace=True)

        return olist_Spenders[:20]

    def recommendations_cust2vec(self, customer_id):
        """Recommends products using Customer2Vec embeddings."""
        Recommendations_c2v = self.olist_details_all[self.olist_details_all['Customer ID'] == customer_id]
        return Recommendations_c2v[['Product Name', 'Review Score']]

    @functools.lru_cache(maxsize=None)
    def get_recommendations(self, customer_id):
        """Provides recommendations based on customer segment."""
        result_final = pd.DataFrame()
        if (customer_id not in self.olist_customers and customer_id not in self.all_customer
                and customer_id not in self.olist_all and customer_id not in self.big_spenders
                and customer_id not in self.loyalists):
            result_final = self.get_popular_products()
        elif customer_id in self.big_spenders:
            result_final = self.upsell_products()
        elif customer_id in self.loyalists:
            result_final = self.recommendations_cust2vec(customer_id)
        elif customer_id in self.all_customer:
            result_final = self.recommendations_als(customer_id)
        elif customer_id in self.olist_all:
            result_final = self.recommendations_cust2vec(customer_id)
        return result_final
