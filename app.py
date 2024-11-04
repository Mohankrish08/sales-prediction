# importing libraries
import streamlit as st 
import numpy as np 
import pandas as pd
import streamlit_lottie as st_lottie 
import time
import requests
import pickle
import google.generativeai as genai

# sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# set the page config
st.set_page_config(page_title="sales analysis", page_icon=":rocket:", layout="wide")

# gemini api
genai.configure(api_key="<API KEY>")

model = genai.GenerativeModel(model_name='gemini-1.5-flash', tools="code_execution")
# url loader
def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# functions
def recommendation_prompt(data):
    prompt = f""" 

    data: {data}

    based on the user data, you have provide the suggestion for the customers.

    """
    return prompt

def sentiment_prompt(gender, product_line, sentiment, rating, profile):
    prompt = f""" 

    gender: {gender}

    product line: {product_line}

    sentiment: {sentiment}

    rating: {rating}

    profile: {profile}

    based on the given data, you have provide below, only provide based on the given data!!
    Don't generate apart from the template

    Template:

    sentiment: <sentiment of the customer based on the data>
    
    recommendation: <provide the suggestion based on profile>

    Behaviour analysis: <provide some behaviour analysis of an customer>

    present the data in a good format, don't write a code.

    """
    return prompt

def generate_content(text):
    try:
        response = model.generate_content(text)
        return response
    except Exception as e:
        return e
    
def categorize_rating(rating):
    if rating >= 7:
        return "Positive"
    elif rating > 5:
        return "Neutral"
    else:
        return "Negative"
    
def customer_details(cust_type, product_line, city):
    out = f""" 

    Customer type: {cust_type}
    product line: {product_line}
    city: {city}

    """
    return out

# loading assets
home = loader_url("https://lottie.host/f40844f5-57eb-416f-9176-5458ae83236d/gHpQbR3UXd.json")


st.markdown("<h1 style='text-align: center;'>Sales Prediction</h1>", unsafe_allow_html=True)

st.lottie(home, height=200, key='front')




branch_map = {'Select options': 0, 'A': 1, 'B': 2, 'C': 3}
city_map = {'Select options': 0, 'Mandalay': 1, 'Naypyitaw': 2, 'Yangon': 3}
customer_type_map = {'Select Customer type': 0, 'Member': 1, 'Normal': 2}
gender_map = {'Male': 0, 'Female': 1}
product_line_map = {
    'Select product line': 0,
    'Fashion accessories': 1,
    'Sports and travel': 2,
    'Food and beverages': 3,
    'Electronic accessories': 4,
    'Health and beauty': 5,
    'Home and lifestyle': 6
}
payment_map = {'Ewallet': 1, 'Cash': 2, 'Credit card': 3}

with st.container():
    # Row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        branch = st.selectbox("Branch", options=list(branch_map.keys()))
        branch_encoded = branch_map[branch]
    with col2:
        city = st.selectbox("City", options=list(city_map.keys()))
        city_encoded = city_map[city]
    with col3:
        customer_type = st.selectbox("Customer type", options=list(customer_type_map.keys()))
        customer_type_encoded = customer_type_map[customer_type]
    with col4:
        gender = st.selectbox("Gender", options=list(gender_map.keys()))
        gender_encoded = gender_map[gender]
    with col5:
        product_line = st.selectbox("Product line", options=list(product_line_map.keys()))
        product_line_encoded = product_line_map[product_line]

    # Row 2
    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        unit_price = st.number_input("Unit price", min_value=1)
    with col7:
        quantity = st.number_input("Quantity", min_value=1)
    with col8:
        tax = st.number_input("Tax 5%")
    with col9:
        total = st.number_input("Total")
    with col10:
        payment = st.selectbox("Payment", options=list(payment_map.keys()))
        payment_encoded = payment_map[payment]
    

    # Row 3
    col11, col12, col13, col14 = st.columns(4)
    with col11:
        cogs = st.number_input("Cogs", min_value=1)
    with col12:
        rating = st.slider("Rating", min_value=1, max_value=10)
    with col13:
        year = st.number_input("Year", min_value=1900, max_value=2020)
    with col14:
        month = st.number_input("Month", min_value=1, max_value=12)

st.write("---")

st.markdown("<h4 style='text-align: left;'>Select the model_out</h4>", unsafe_allow_html=True)

model_out = st.selectbox(label="Select a model_out", options=["Select the model_out","Linear Regression","Random Forest", "Decision Tree", "SVM", ])
        
nn = st.checkbox("Run")

if nn:

    data_out = [branch_encoded, city_encoded, customer_type_encoded, gender_encoded, product_line_encoded, unit_price, quantity, tax, total, payment_encoded, cogs, 4.761905, rating, year, month]

    data_out_arr = np.array(data_out).reshape(1, -1)


    if model_out == "Linear Regression":
        with open("./pickle/linear.pkl", "rb") as f:
            model_out = pickle.load(f)

            output = model_out.predict(data_out_arr)

            st.write(output)

    elif model_out == "Random Forest":
        with open("./pickle/rf.pkl", "rb") as f:
            model_out = pickle.load(f)

            output = model_out.predict(data_out_arr)

            st.write(output)

    elif model_out == "Decision Tree":
        with open("./pickle/tree.pkl", "rb") as f:
            model_out = pickle.load(f)

            output = model_out.predict(data_out_arr)

            st.write(output)

    elif model_out == "SVM":
        with open("./pickle/svm.pkl", "rb") as f:
            model_out = pickle.load(f)

            output = model_out.predict(data_out_arr)

            st.write(output)

    st.write("--")

    st.markdown("<h4 style='text-align: left;'>Analysis</h4>", unsafe_allow_html=True)

    st.write(generate_content(sentiment_prompt(gender, product_line, categorize_rating(rating), rating, profile=(customer_details(customer_type, product_line, city)))).text)

    st.write('---')

    
