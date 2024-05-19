import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from recommendation_utils import RecommendationSystem
from viz_utils import *

raw_path = '../../Data Source/'
olist = pd.read_csv(raw_path + 'olist_master.csv')

total_customers = int(len(olist['customer_id'].unique()))
total_orders = int(len(olist['order_id'].unique()))
total_products = int(len(olist['product_id'].unique()))
total_revenue = olist['price'].sum()


def product_and_customer_behavior_analysis():
    st.subheader('Product & Customer Behaviour Analysis')
    st.success("Knowing who your Customers are is great, but knowing how they behave is even better.")

    # 1. Orders count across days of week and hourly comparison (heatmap)
    purchase_count = olist.groupby(['order_purchase_day', 'order_purchase_hour']).nunique()['order_id'].unstack()
    st.markdown('**Số lượng đơn hàng theo ngày trong tuần và giờ.**')
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(purchase_count.reindex(index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                cmap="YlGnBu", annot=True, fmt="d", linewidths=0.2, ax=ax)
    ax.set_xlabel('Purchases/Hour')
    ax.set_ylabel('Day of Week')

    st.write(fig)

    # Add comments
    st.markdown("""
    Bản đồ nhiệt trực quan hóa số lượng đơn đặt hàng được đặt vào mỗi ngày trong tuần và giờ trong ngày. Màu xanh càng đậm thì số lượng đơn hàng càng cao.

    Xu hướng chung:
    - Giờ bận rộn nhất cho các đơn đặt hàng dường như là từ 10 giờ sáng đến 2 giờ chiều liên tục trong suốt cả tuần. Điều này có thể là do mọi người duyệt và mua hàng trong giờ nghỉ trưa hoặc trong giờ nghỉ làm giữa buổi sáng.
    - Thứ Ba, Thứ Tư và Thứ Năm dường như có tổng số lượng đơn đặt hàng cao nhất, đặc biệt là trong những giờ cao điểm nêu trên. Thứ Hai và thứ Sáu cũng có lượng đặt hàng đáng kể, mặc dù thấp hơn một chút so với những ngày giữa tuần.
    - Thứ Bảy và Chủ Nhật chứng kiến số lượng đơn đặt hàng thấp nhất, cho thấy hoạt động mua sắm trực tuyến có thể sụt giảm trong những ngày cuối tuần.
    """)

    # 2. Sales by Month & Day of Week (line chart)
    sales_per_purchase_month = olist.groupby(['order_purchase_month', 'order_purchase_mon', 'order_purchase_day'],
                                             as_index=False).payment_value.sum()
    sales_per_purchase_month = sales_per_purchase_month.sort_values(by=['order_purchase_month'], ascending=True)
    fig = px.line(sales_per_purchase_month, x="order_purchase_mon", y="payment_value", color='order_purchase_day')
    fig.update_layout(title="Doanh số theo tháng và ngày trong tuần.",
                      xaxis_title="Tháng",
                      yaxis_title="Doanh thu (USD)",
                      font=dict(family="Courier New, monospace", size=15, color="#7f7f7f"))
    st.write(fig)

    st.markdown("""
    Biểu đồ cung cấp những hiểu biết có giá trị về xu hướng bán hàng qua các tháng và ngày khác nhau trong tuần.
    
    Xu hướng chung:
    - Tổng số liệu bán hàng cho thấy sự biến động đáng kể trong suốt cả năm, cho thấy tính thời vụ tiềm ẩn hoặc các yếu tố ảnh hưởng khác. Các tháng như tháng 9 và tháng 10 cho thấy doanh số bán hàng tương đối thấp hơn, trong khi tháng 11 và tháng 12 chứng kiến mức tăng mạnh, có thể do hoạt động mua sắm trong dịp lễ.
    - Có sự khác biệt rõ ràng về doanh số bán hàng giữa các ngày khác nhau trong tuần, với một số ngày luôn hoạt động tốt hơn những ngày khác.
    Quan sát cụ thể:
    - Các ngày trong tuần thường có doanh số bán hàng cao hơn so với cuối tuần, trong đó Thứ Hai, Thứ Ba và Thứ Tư thường cho thấy hiệu suất cao hơn. Điều này cho thấy phần lớn hoạt động mua hàng diễn ra trong tuần làm việc.
    - Doanh số bán hàng giảm đáng kể vào cuối tuần, đặc biệt là vào Chủ nhật. Thứ Bảy dường như có doanh số bán hàng cao hơn một chút so với chủ nhật nhưng vẫn thấp hơn các ngày trong tuần.
    """)

    # 3. Customer Rating Each Month & Day of Week (line chart)
    olist['review_answer_timestamp'] = pd.to_datetime(olist['review_answer_timestamp'])
    olist['review_dayofweek'] = olist.review_answer_timestamp.dt.dayofweek
    olist['review_day'] = olist['review_dayofweek'].map(
        {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
    olist['review_month'] = olist.review_answer_timestamp.dt.month.map(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov',
         12: 'Dec'})
    review_score_per_month = olist.groupby(['review_month', 'review_day'], as_index=False).review_score.mean()
    fig = px.line(review_score_per_month, x="review_month", y="review_score", color='review_day',
                  title='Đánh giá của khách hàng theo từng tháng và ngày trong tuần')
    fig.update_layout(title="Ratings Each Month & Day of Week",
                      xaxis_title="Tháng ",
                      yaxis_title="Đánh giá xếp hạng",
                      font=dict(family="Courier New, monospace", size=15, color="#7f7f7f"))
    st.write(fig)
    st.markdown("""
    Xu hướng hàng tháng:
    - **Tháng 4-Tháng 8**: Khoảng thời gian từ tháng 4 đến tháng 8 cho thấy điểm đánh giá tương đối cao hơn với một số đỉnh và thung lũng.
    - **Tháng 9-Tháng 3**: Các tháng từ tháng 9 đến tháng 3 thể hiện xu hướng không nhất quán hơn với điểm đánh giá giảm xuống thấp hơn.
    - **Điểm thấp trong tháng 3**: Tháng 3 cho thấy điểm đánh giá giảm đáng kể trong hầu hết các ngày trong tuần, cho thấy các vấn đề tiềm ẩn trong giai đoạn này cần được điều tra thêm.
    """)

    # 4. Average Review Ratings across Product Categories (bar chart)
    avg_score_per_category = olist.groupby('product_category_name', as_index=False).agg(
        {'review_score': ['count', 'mean']})
    avg_score_per_category.columns = ['Product Category', 'Number of Reviews', 'Average Review Ratings']
    avg_score_per_category = avg_score_per_category[avg_score_per_category['Number of Reviews'] > 100]
    avg_score_per_category = avg_score_per_category.sort_values(by='Number of Reviews', ascending=False)
    avg_ratings = avg_score_per_category[:20]
    fig = px.bar(avg_ratings, x='Product Category', y='Number of Reviews',
                 hover_data=['Average Review Ratings'], color='Average Review Ratings', height=500,
                 title='Đánh giá trung bình trên các loại sản phẩm')
    fig.update_layout(
        xaxis_title="Danh mục sản phẩm",
        yaxis_title="Số lượng đánh giá",
    )
    st.write(fig)
    st.markdown("""
    - Biểu đồ được sắp xếp theo thứ tự giảm dần về số lượng đánh giá, trong đó "Bed Bath & Table" có số lượng đánh giá cao nhất và "Consoles Games" có số lượng đánh giá thấp nhất trong số 20 danh mục hàng đầu. Điều này cho thấy mức độ tương tác và mức độ phổ biến khác nhau của khách hàng đối với các loại sản phẩm khác nhau.
    
    Quan sát cụ thể:
    - Số lượng đánh giá cao, Xếp hạng trung bình cao: Các danh mục như "Bed Bath & Table", "Health & Beauty" và "Sports & Leisure" có số lượng đánh giá cao và xếp hạng trung bình tương đối cao (trên 4,0). Điều này cho thấy một cơ sở khách hàng lớn với những trải nghiệm nhìn chung tích cực.
    - Số lượng đánh giá vừa phải, Xếp hạng trung bình vừa phải: Một số danh mục, bao gồm "Furniture & Decor", "Computers & Accessories", "Watches & Gifts", và "Toys" nằm trong phạm vi giữa của cả số lượng đánh giá và xếp hạng trung bình.
    - Số lượng đánh giá thấp, xếp hạng trung bình đa dạng: Các danh mục như "Electronics", "Stationery", và "Pet Shop" có số lượng đánh giá thấp hơn nhưng xếp hạng trung bình của chúng khác nhau, trong đó "Electronics" có xếp hạng trung bình cao hơn so với các danh mục khác.
    - Số lượng đánh giá thấp nhất, xếp hạng trung bình thấp hơn: "Console Games" có số lượng đánh giá thấp nhất trong số 20 danh mục hàng đầu và xếp hạng trung bình tương đối thấp hơn.
    """)

    # 5. Product-Wise Revenue (sunburst charts)
    total_rev_month = olist.groupby(['order_purchase_year', 'order_purchase_mon', 'product_category_name'],
                                    as_index=False).payment_value.sum()
    total_rev_month.columns = ['Sales Year', 'Sales Month', 'Product Category', 'Sales Revenue']
    top_products = total_rev_month.groupby('Product Category')['Sales Revenue'].sum().sort_values(ascending=False).head(
        10).index

    # Lọc dữ liệu cho 10 sản phẩm hàng đầu
    top_data = total_rev_month[total_rev_month['Product Category'].isin(top_products)]

    top_data = top_data.sort_values(by=['Sales Year', 'Product Category'])

    fig = px.bar(
        top_data,
        x="Product Category",
        y="Sales Revenue",
        color="Sales Year",
        barmode="group",  # Hiển thị cột ghép
        title='Top 10 sản phẩm có tổng doanh thu cao nhất trong 3 năm'
    )

    # Tùy chỉnh bố cục
    fig.update_layout(
        xaxis_title="Loại sản phẩm",
        yaxis_title="Doanh thu",
        xaxis=dict(tickangle=-45),  # Xoay nhãn trục x
    )

    # Hiển thị chú thích (legend) ở bên phải
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1))

    # Hiển thị giá trị trên mỗi cột
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')

    # Hiển thị biểu đồ
    st.write(fig)

    st.markdown("""
    Quan sát tổng thể:
    - Danh mục chiếm ưu thế: "Bed Bath & Table" và "Health & Beauty" liên tục chiếm ưu thế doanh số bán hàng trong cả năm 2017 và 2018, chiếm phân khúc lớn nhất trong cả hai năm.
    - Tăng trưởng và Suy thoái: Trong khi một số danh mục duy trì hiệu suất tương đối ổn định trong hai năm, những danh mục khác lại có sự tăng trưởng hoặc suy giảm đáng chú ý. Ví dụ: "Computers & Accessories" và "Furniture & Decor" dường như đã tăng thị phần trong năm 2018 so với năm 2017.
    Quan sát cụ thể:
    - Thành tích năm 2017: "Bed Bath & Table" chiếm thị phần lớn nhất trong năm 2017, tiếp theo là "Health & Beauty", "Computers & Accessories", và "Furniture & Decor". Các danh mục như "Sports & Leisure", "Watches & Gifts", và "Housewares" cũng đóng góp đáng kể.
    - Xu hướng năm 2018: “Bed Bath & Table” duy trì vị trí dẫn đầu trong năm 2018, trong khi "Health & Beauty" tăng trưởng nhẹ. "Computers & Accessories", và "Furniture & Decor" có thị phần tăng rõ rệt hơn, cho thấy tiềm năng tăng trưởng về nhu cầu đối với các danh mục này.
    """)

    # 6. Product Wise Revenue by Hour (sunburst chart)
    total_rev_hour = olist[olist['order_purchase_year'] == 2018].groupby(
        ['order_purchase_hour', 'product_category_name'], as_index=False).payment_value.sum()
    total_rev_hour.columns = ['Sales Hour', 'Product Category', 'Sales Revenue']
    # st.markdown('Product Wise **_Revenue_ _by_ Hour**')
    # fig = px.sunburst(total_rev_hour, path=['Sales Hour', 'Product Category'], values='Sales Revenue',
    #                   color='Sales Revenue', hover_data=['Product Category'],
    #                   color_continuous_scale='RdBu',
    #                   color_continuous_midpoint=np.average(total_rev_hour['Sales Revenue'],
    #                                                        weights=total_rev_hour['Sales Revenue']))
    # st.write(fig)

    # Tính tổng doanh thu cho mỗi Product Category và chọn top 10
    product_revenue = total_rev_hour.groupby("Product Category")["Sales Revenue"].sum()
    top_n_products = product_revenue.nlargest(10).index

    # Lọc dữ liệu chỉ lấy các sản phẩm trong top 10
    filtered_data = total_rev_hour[total_rev_hour['Product Category'].isin(top_n_products)]

    # Vẽ biểu đồ cột
    fig = px.bar(
        filtered_data,
        x="Sales Hour",
        y="Sales Revenue",
        color="Product Category",
        title='Doanh thu theo giờ của top 10 sản phẩm bán chạy nhất (2018)'
    )

    # Cập nhật bố cục
    fig.update_layout(
        xaxis_title="Giờ mua hàng",
        yaxis_title="Doanh thu"
    )

    # Hiển thị biểu đồ
    st.write(fig)
    st.markdown("""
    Giờ cao điểm:
    Mặc dù không có giờ duy nhất nhưng một số giờ dường như có đóng góp doanh thu cao hơn một chút:
    - Chiều đến tối: Các giờ như 15, 16 và 14 dường như chiếm tỷ lệ lớn hơn một chút, có khả năng cho thấy hoạt động của khách hàng đạt đỉnh điểm vào cuối giờ chiều và đầu giờ tối.
    - Giờ ban đêm: Những giờ như 22 và 23 cũng có tỷ lệ phần trăm cao hơn một chút, điều này có thể gợi ý một phân khúc khách hàng thích mua sắm vào đêm khuya.
    """)

    # 7. Product Wise Revenue by Day of Week (sunburst chart)
    total_rev_dayofweek = olist[olist['order_purchase_year'] == 2018].groupby(
        ['order_purchase_day', 'product_category_name'], as_index=False).payment_value.sum()
    total_rev_dayofweek.columns = ['Sales DayofWeek', 'Product Category', 'Sales Revenue']
    # st.markdown('**_Product_ _Wise_ Revenue** by Day of Week')
    # fig = px.sunburst(total_rev_dayofweek, path=['Sales DayofWeek', 'Product Category'], values='Sales Revenue',
    #                   color='Sales Revenue', hover_data=['Product Category'],
    #                   color_continuous_scale='RdBu',
    #                   color_continuous_midpoint=np.average(total_rev_dayofweek['Sales Revenue'],
    #                                                        weights=total_rev_dayofweek['Sales Revenue']))
    # fig.update_layout()
    # st.write(fig)

    # Tìm top N sản phẩm có doanh thu cao nhất
    top_n_products = total_rev_dayofweek.groupby("Product Category")["Sales Revenue"].sum().nlargest(
        10).index  # Ví dụ: lấy top 10 sản phẩm

    # Lọc dữ liệu chỉ lấy các sản phẩm trong top N
    filtered_data = total_rev_dayofweek[total_rev_dayofweek['Product Category'].isin(top_n_products)]

    st.markdown('**_Product_ _Wise_ Revenue** by Day of Week')
    fig = px.bar(
        filtered_data,  # Sử dụng dữ liệu đã lọc
        x="Sales DayofWeek",
        y="Sales Revenue",
        color="Product Category",
        title=f'Doanh thu theo ngày trong tuần của top {len(top_n_products)} sản phẩm có doanh thu cao nhất (2018)'
    )

    # Tùy chỉnh bố cục
    fig.update_layout(
        xaxis_title="Ngày trong tuần",
        yaxis_title="Doanh thu",
        xaxis={'categoryorder': 'array',
               'categoryarray': ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']}
    )

    st.write(fig)


@st.cache_resource
def customer_segmentation_and_ltv():
    st.subheader('Customer Segmentation & LTV')
    st.success("Don't find Customers for your Product, find Products for your Customers")

    # 1. Average Gap between First and Last Purchase (bar chart)
    df_days_repurchase_subsegment_2018 = pd.read_csv('./data/df_days_repurchase_subsegment_2018.csv')
    trace0 = go.Bar(
        x=df_days_repurchase_subsegment_2018["sub_segment"].values,
        y=df_days_repurchase_subsegment_2018["diff_order_purchase"].values,
        marker=dict(color=['rgba(36,123,160,1)', 'rgba(75,147,177,1)', 'rgba(112,193,179,1)',
                           'rgba(138,204,192,1)', 'rgba(243,255,189,1)', 'rgba(247,255,213,1)',
                           'rgba(255,22,84,1)']),
    )
    data = [trace0]
    layout = go.Layout(title='Số ngày trung bình giữa lần mua đầu tiên và lần mua cuối cùng')
    fig = go.Figure(data=data, layout=layout)
    st.write(fig)

    st.markdown("""
    Phân tích số ngày trung bình giữa các lần mua hàng cho thấy các mô hình tương tác khách hàng khác biệt. 
    - Khách hàng "Inactive" thể hiện khoảng cách lớn nhất, làm nổi bật nhu cầu về các nỗ lực tái kích hoạt. Các phân khúc "Hot" hiển thị khoảng cách vừa phải, cho thấy sự kết hợp giữa khách hàng mới và khách hàng quay lại, trong khi các phân khúc "Cold" thể hiện tần suất mua hàng thường xuyên hơn, cho thấy các chiến lược tương tác thành công. 
    - Các phân khúc "Active", như mong đợi, cho thấy khoảng cách ngắn nhất, phản ánh mức độ tương tác cao hơn của họ. 
    """)

    # 2. RFM Customer Segmentation (treemap)
    rfm_level_ag = pd.read_csv('./data/rfm_level_ag.csv')
    fig1 = go.Figure(go.Treemap(
        labels=rfm_level_ag['Customer Segment'],
        parents=['Customer Segmentation'] * len(rfm_level_ag),  # Create parent level for all segments
        values=rfm_level_ag['Monetary.1']
    ))
    fig1.update_layout(title='Phân khúc khách hàng RFM')
    st.write(fig1)

    st.markdown("""
    1. **VVIP - Can't Lose Them:**
    Phân khúc bao gồm các người dùng có giá trị nhất với điểm RFM cao nhất. Họ mua hàng thường xuyên, gần đây và có mức chi tiêu đáng kể
    2. **Champions Big Spenders:**
    Người dùng trong phân khúc này có thói quen chi tiêu cao và mua hàng thường xuyên, nhưng độ mới (recency) của họ có thể thay đổi.
    3. **Loyal Customers:**
    Phân khúc bao gồm các người dùng mua hàng thường xuyên với mức chi tiêu vừa phải. 
    4. **Potential Loyalists:**
    Những người dùng cho thấy tiềm năng trở thành khách hàng trung thành, thể hiện điểm frequency và recency tốt nhưng vẫn có cơ hội để tăng trưởng monetary value.
    5. **Needs Attention:**
    Phân khúc bao gồm các người dùng đã mua hàng nhưng gần đây không hoạt động. Họ có nguy cơ chuyển nhà cung cấp dịch vụ.
    6. **Hibernating - Almost Lost:**
    Những người dùngày đã mua hàng trong quá khứ nhưng không hoạt động trong một thời gian dài. Việc thu hút lại họ có thể cần tốn nhiều công sức hơn.
    7. **Lost Customers:**
    Phân khúc đại diện cho các người dùng có điểm RFM thấp nhất, cho thấy mua hàng không thường xuyên, chi tiêu thấp và thời gian không hoạt động lâu.
    """)

    # 3. K-Means based Clusters (heatmap)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown('**_K-Means_ _based_ Clusters**')
    relative_imp = pd.read_csv('./data/rel_imp.csv')
    plt.figure(figsize=(13, 5))
    plt.title('Relative importance of attributes')
    fig2 = sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
    plt.show()
    st.pyplot()
    st.write(fig2)
    st.markdown("""
    - **Cluster 0:** Nhóm khách hàng đóng góp nhiều nhất vào doanh thu. Họ mua sắm thường xuyên, chi tiêu hào phóng và đã mua hàng gần đây. Đây là phân khúc có giá trị nhất và là nhóm khách hàng mà chúng ta cần tập trung.
    - **Cluster 1:** Những khách hàng có mức độ Recency (mức độ mua hàng gần đây), Frequency (tần suất mua hàng) và Monetary Value (giá trị tiền tệ) thấp. Cần thiết kế các chiến dịch nhắm mục tiêu để thu hút họ mua sắm trở lại.
    - **Cluster 2:** Những khách hàng đã mua sắm gần đây, nhưng mức chi tiêu và tần suất mua hàng của họ thấp hơn mong muốn. Việc đề xuất sản phẩm phù hợp có thể giúp họ khám phá các mặt hàng mới, tăng giá trị đơn hàng và tối đa hóa giá trị của họ theo thời gian.
    - **Cluster 3:** Nhóm có lịch sử chi tiêu khá tốt, nhưng việc họ không hoạt động trong vài tháng qua là điều đáng lo ngại. Để ngăn người dùng chuyển sang đối thủ cạnh tranh, cần can thiệp nhanh chóng với các ưu đãi và khuyến mãi chủ động để thu hút họ quay trở lại.
    """)

    # 4. Customer Lifetime Value (images)
    # st.markdown('**_Customer_ _Lifetime_ Value**')
    # image_files = [
    #     './images/conditional_expected_average_profit.png',
    #     './images/frequency_recency_matrix.png',
    #     './images/plot_history_alive.png',
    #     './images/probability_alive_matrix.png'
    # ]
    # captions = [
    #     'Conditional expected average profit',
    #     'Frequency-Recency Matrix',
    #     'History alive',
    #     'Probability'
    # ]
    # for image_file, caption in zip(image_files, captions):
    #     image = Image.open(image_file)
    #     st.image(image, caption=caption, use_column_width=True)


def geography_analysis():
    # st.subheader('Geography Analysis')
    # st.success("Social Media is a whole new world, that's where the opportunity resides. Are you Analysing it?")

    st.markdown('**Đánh giá của khách hàng trên khắp Brazil**')
    image8 = Image.open('./images/mapplot.png')
    st.image(image8, use_column_width=True)
    st.markdown("""Bản đồ mật độ mã zip của Brazil cho thấy sự phân bố không đồng đều, với các cụm mật độ cao tập trung tại khu vực Đông Nam (São Paulo, Rio de Janeiro, Minas Gerais), khu vực phía Nam và dọc theo bờ biển Đông Bắc, phản ánh sự đô thị hóa và phát triển kinh tế tại những vùng này. Ngược lại, vùng Amazon và khu vực Trung Tây thể hiện mật độ thấp hơn do đặc điểm dân cư thưa thớt và kinh tế kém phát triển. Sự tập trung mã zip dọc theo bờ biển cũng làm nổi bật mô hình phát triển lịch sử và kinh tế của Brazil, nơi các thành phố ven biển đóng vai trò trung tâm thương mại và dân cư.
    """)

    st.markdown('**Thời gian giao hàng trung bình**')
    image9 = Image.open('./images/average_delivery.png')
    st.image(image9, use_column_width=True)
    st.markdown("""

    """)
    st.markdown('**Thương mại điện tử ở bang Brazil**')
    image10 = Image.open('./images/best_state.png')
    st.image(image10, use_column_width=True)
    st.markdown("""
    - Phân chia vận chuyển hàng hóa và giao hàng: Các bang như RR và AC phải đối mặt với chi phí vận chuyển hàng hóa cao và thời gian giao hàng dài, có thể do những thách thức về địa lý và cơ sở hạ tầng. Ngược lại, các bang như SP và MG có chi phí thấp hơn và giao hàng nhanh hơn nhờ vị trí trung tâm và mạng lưới hậu cần phát triển.
    - Những bất ngờ về giao hàng: Điều thú vị là một số tiểu bang có thời gian giao hàng dài hơn (AP, RR, AC) luôn thực hiện tốt hơn ước tính thời gian giao hàng, trong khi những tiểu bang khác (CE, ES) lại có thời gian ước tính giao hàng lâu hơn.
    """)


def recommendation():
    # Initialize RecommendationSystem object
    recommender = RecommendationSystem()

    customer_segments_df = recommender.get_customer_segments()
    st.header("Product Recommendation System")

    st.dataframe(customer_segments_df.style.format({'Customer ID': '{}'}), hide_index=True, use_container_width=True)

    customer_id = st.text_input("Enter Customer ID:")

    if customer_id:
        try:
            customer_id = int(customer_id)

            result_final = recommender.get_recommendations(customer_id)
            if not result_final.empty:
                st.write(f"### Recommendations for customer {customer_id}:")
                st.dataframe(result_final, hide_index=True, use_container_width=True)
            else:
                st.write("No recommendations found for this customer.")

        except ValueError:
            st.write("Invalid customer ID. Please enter a number.")


def highlight_positive(val):
    color = 'green' if val == 'Positive' else 'black'
    return f'color: {color}'


def sentiment():
    # Positive Reviews:
    # "This product exceeded my expectations! It's exactly what I needed."
    # "I'm very satisfied with this purchase. The quality is excellent and it arrived quickly."
    # "I love this product! It's easy to use and has made my life so much easier."
    # "Great value for the price. I would highly recommend it to others."
    # "Outstanding customer service! They were very helpful and responsive to my inquiries."

    # Negative Reviews:
    # "I'm very disappointed with this product. It broke after just a few uses."
    # "The quality of this item is poor. It doesn't work as advertised."
    # "This was a waste of money. It didn't meet my expectations at all."
    # "Terrible experience with this company. They were unresponsive and unhelpful."
    # "I regret buying this product. It's cheaply made and doesn't function properly."

    # Load the sentiment predictions CSV file
    predictions = pd.read_csv("./data/sentiment_predictions.csv")
    predictions = predictions.dropna(subset=['review'])

    # Xóa các dòng trùng lặp dựa trên cột 'review'
    predictions = predictions.drop_duplicates(subset=['review'])

    # Tính số lần xuất hiện của mỗi product_id
    product_counts = predictions['product_id'].value_counts()

    # Sắp xếp product_id theo thứ tự giảm dần
    sorted_product_ids = product_counts.index.tolist()

    # Thay đổi cách hiển thị
    st.header("Search Reviews by Product ID")
    selected_product_id = st.selectbox("Select Product ID:", sorted_product_ids)

    # Lọc và hiển thị đánh giá dựa trên product_id đã chọn
    if selected_product_id:
        filtered_reviews = predictions[predictions['product_id'] == selected_product_id][
            ['review', 'sentiment', 'probability']]
        if not filtered_reviews.empty:
            st.markdown(f"### Reviews for ID: {selected_product_id}")
            st.dataframe(filtered_reviews.style.applymap(highlight_positive, subset=['sentiment']),
                         use_container_width=True)
        else:
            st.write("No reviews found for the selected Product ID.")

    # Allow users to search for reviews by text input
    # st.header("Sentiment Review ")
    # user_text = st.text_area("Enter Text:")
    # if user_text:
    #     filtered_reviews = predictions[predictions['review'].str.contains(user_text, case=False, na=False)][
    #         ['review', 'sentiment']]
    #     if not filtered_reviews.empty:
    #         st.subheader("Filtered Reviews")
    #         st.dataframe(filtered_reviews.style.applymap(highlight_positive, subset=['sentiment']),
    #                      use_container_width=True)
    #     else:
    #         st.write("No reviews found for the entered text.")


# Main Streamlit App
def main():
    menu = ["Product & Customer Behaviour Analysis", "Geography Analysis"
            , "Customer Segmentation & LTV","Recommendation System", "Review Sentiment"]
    choices = st.selectbox("Select Analytics Dashboard", menu)

    st.title("Analytics Dashboard!")
    st.markdown("""<style>body {    color: #fff;    background-color: #0A3648;}</style>    """, unsafe_allow_html=True)

    if choices == 'Product & Customer Behaviour Analysis':
        df = pd.DataFrame({
            'Total Customers': [total_customers],
            'Total Orders': [total_orders],
            'Total Products': [total_products],
            'Total Revenue': [total_revenue],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)
        product_and_customer_behavior_analysis()
    elif choices == 'Customer Segmentation & LTV':
        customer_segmentation_and_ltv()
    elif choices == 'Geography Analysis':
        geography_analysis()
    elif choices == 'Recommendation System':
        recommendation()
    elif choices == 'Review Sentiment':
        sentiment()


if __name__ == '__main__':
    main()
