import numpy as np
import pandas as pd
import pickle
import streamlit as st
import pickle
from underthesea import word_tokenize, pos_tag, sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read data
df = pd.read_csv('ThoiTrangNam_raw_cleaned.csv', encoding='utf8')
df2 = pd.read_csv("data_rating_cleaned.csv")

# load model and others:
with open('surprise_svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
# load tfidf
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
# load index
with open('index.pkl', 'rb') as f:
    index = pickle.load(f)

#--------------
# GUI
st.title("Customer Recommendation")
# Upload file
#uploaded_file = st.file_uploader("Choose a file", type=['csv'])
#if uploaded_file is not None:
    #data = pd.read_csv(uploaded_file, encoding='latin-1')
    #data.to_csv("spam_new.csv", index = False)


# GUI
menu = ["Introduction", "Data infomation", "Recommedation by user", "Recommedation by description" ]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Introduction':
    st.subheader("Introduction")
    st.write("""
    ###### Customer Recommendation System (Hệ thống đề xuất khách hàng) là một hệ thống có khả năng đưa ra các đề xuất sản phẩm, dịch vụ hoặc nội dung phù hợp với từng khách hàng cá nhân dựa trên hành vi, sở thích và dữ liệu liên quan của họ. Bài này sử dụng dữ liệu thời trang nam trên Shoppee sau khi đã được pre-processing.
    """)  
    st.write("""###### Problem/ Requirement: Project này sử dụng Machine Learning algorithms Surprise và Gensim trong hệ thống đề xuất sản phẩm theo người dùng (user) và theo mô tả sản phẩm.""")
    st.image("recommendation-system.png")
    st.write(
        """###### Dữ liệu gồm 2 file: tạm gọi là 'fashion' và 'rating'.""")

elif choice == 'Data infomation':
    st.subheader("View some data infomation")
    st.write("##### 1. Some data on 'fashion' file:")
    st.dataframe(df.head(3))
    st.write("###### sub category most selling:")
    st.image("sub_category.png")
    st.write("##### 2. Some data on 'rating' file:")
    st.dataframe(df2.tail(3))
    st.write("###### 10 products most rating:")
    st.image("10_products.png")


elif choice == 'Recommedation by user':
    data = df2[['user_id', 'user']]
    df_sample = data.sample(5)
    # In 3 khách hàng này ra màn hình
    st.write("Danh sách khách hàng ngẫu nhiên:")
    st.write(df_sample)
    st.write("##### Đề xuất sản phẩm cho khách hàng:")

    # Tạo một điều khiển và đưa khách hàng ngẫu nhiên này vào đó
    user = st.selectbox("Chọn khách hàng:", df_sample['user'])
    st.write("Khách hàng đã chọn:", user)
    user_id = df2[df2['user'] == user]['user_id'].values[0]
    st.write("Thông tin Đề xuất sản phẩm cho khách hàng đã chọn:")

    # function: in ra 5 sản phẩm có EstimateScore>=3 lớn nhất
    def get_recommendation(user_id):
        df_select = df2[(df2['user_id'] == user_id) & (df2['rating'] >= 3)]
        # df_select = df_select.set_index('product_id')
        df_score = df2[["product_id", 'user_id', 'user', 'rating', 'product_name']]
        df_score['EstimateScore'] = df_score.apply(lambda x: model.predict(x['user_id'], x['product_id']).est, axis=1)
        # df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: algorithm.predict(user_id, x).est) # est: get EstimateScore
        df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
        # loại bỏ duplicated values
        df_score = df_score.drop_duplicates()
        return df_score[df_score.EstimateScore >= 3].head()
    # nhập vào mã người dùng (user_id:", in ra 5 sản phẩm có recommendation cao nhất
    df_recommendation = get_recommendation(user_id)
    df_recommendation


elif choice == 'Recommedation by description':
    def recommend_product_by_name(search):  # , dictionary, tfidf, index
        # Preprocess the product name
        search = word_tokenize(search, format="text")
        print("product_name:", search)
        # Convert search words into sparse vectots
        search = search.lower().split()
        kw_vector = dictionary.doc2bow(search)
        print("View product's vector:")
        print(kw_vector)
        # Similar calculation
        sim = index[tfidf[kw_vector]]
        # print result:
        list_id = []
        list_score = []
        for i in range(len(sim)):
            list_id.append(i)
            list_score.append(sim[i])
        df_result = pd.DataFrame({'id': list_id,
                                  'score': list_score})

        # five highest scores
        five_highest_score = df_result.sort_values(by='score', ascending=False).head(6)
        print("Five highest score: ")
        print(five_highest_score)
        print("Ids to list: ")
        idToList = list(five_highest_score['id'])
        print(idToList)

        products_find = df[df.index.isin(idToList)]
        results = products_find[["product_id", "product_name"]]
        results = pd.concat([results, five_highest_score], axis=1).sort_values(by="score", ascending=False)
        return results

    search = st.text_input("Nhập thông tin tìm kiếm:")
    result = recommend_product_by_name(search)
    # In danh sách sản phẩm tìm được ra màn hình
    st.write("Danh sách sản phẩm tìm được:")
    st.dataframe(result)


