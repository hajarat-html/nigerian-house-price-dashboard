# Data Cleaning & Manipulation libraries
import pandas as pd
import numpy as np

# Data App library
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.chart_container import chart_container

# Machine Learning models library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data Visualization library
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# To handle file & tasks
import os
import joblib

# Set up the page configuration
page_title = "Nigerian House Price Dashboard"
page_icon = ":house:"
layout = "centered"

# Set up the page configuration
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f6; /* Light gray background */
    }

    /* Main content area padding */
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar styling */
    .css-1lcbmhc, .css-1d391kg { /* Targeting sidebar elements for rounded corners and shadow */
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 15px;
        background-color: #ffffff; /* White background for sidebar */
    }

    /* Title styling */
    h1 {
        color: #2c3e50; /* Dark blue-gray */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Subheader styling */
    h2, h3, h4, h5, h6 {
        color: #34495e; /* Slightly lighter dark blue-gray */
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }

    /* Info boxes (st.info) */
    .stAlert.info {
        background-color: #e0f2f7; /* Light blue */
        border-left: 8px solid #2196f3; /* Blue border */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Success boxes (st.success) */
    .stAlert.success {
        background-color: #e8f5e9; /* Light green */
        border-left: 8px solid #4caf50; /* Green border */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Warning boxes (st.warning) */
    .stAlert.warning {
        background-color: #fff3e0; /* Light orange */
        border-left: 8px solid #ff9800; /* Orange border */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        overflow: hidden; /* Ensures rounded corners apply to content */
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    /* Selectbox/Multiselect */
    .stSelectbox>div>div, .stMultiSelect>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Sliders */
    .stSlider .st-bb { /* Track */
        background-color: #d1e7dd; /* Light green track */
        border-radius: 5px;
    }
    .stSlider .st-bd { /* Thumb */
        background-color: #4CAF50; /* Green thumb */
        border: 2px solid #4CAF50;
        border-radius: 50%;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa; /* Light background for expander header */
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .streamlit-expanderContent {
        border-left: 3px solid #4CAF50; /* Green line on the left of content */
        padding-left: 15px;
        margin-left: 5px;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    /* Custom container for home page sections */
    .home-section-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
        border: 1px solid #e0e0e0;
    }

    /* Specific styling for the main title on Home page */
    .main-title-home {
        font-size: 3em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Custom styling for the overview and pages section on Home */
    .overview-box, .pages-box {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }
    .overview-box p, .pages-box ul {
        font-size: 1.1em;
        line-height: 1.6;
        color: #555;
    }
    .pages-box ul li {
        margin-bottom: 0.5em;
    }

</style>
""", unsafe_allow_html=True)

# Navigation Menu
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Home", "Dataset", "Visualizations", "3D Visualizations", "Filtered Data",
                 "Features", "Data Exploration", "Geospatial Analysis", "Model", "Prediction"],
        icons=["house-door-fill", "table", "bar-chart-fill", "cube", "filter-square",
               "clipboard-data", "search", "geo-alt", "cpu-fill", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )


# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    file_path = "nigeria_houses_data.csv"
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the dataset is in the same directory as app.py.")
        return pd.DataFrame() # Return empty DataFrame on error

    # Feature Engineering: Create 'total_rooms'
    data['total_rooms'] = data['bedrooms'] + data['bathrooms'] + data['toilets'] + data['parking_space']

    # Initial Outlier Trimming: Based on price quantiles (0.15 and 0.85)
    q1_price, q9_price = data["price"].quantile([0.15, 0.85])
    mask_price = data["price"].between(q1_price, q9_price)
    trimmed_data = data[mask_price].copy()

    # Remove states with few entries (less than 100 entries)
    records = trimmed_data['state'].value_counts()
    trimmed_data = trimmed_data[~trimmed_data['state'].isin(records[records < 100].index)].copy()

    # Ensure numerical columns are numeric, coerce errors to NaN and then drop for safety
    numerical_cols = ['bedrooms', 'bathrooms', 'toilets', 'parking_space', 'price', 'total_rooms']
    for col in numerical_cols:
        trimmed_data[col] = pd.to_numeric(trimmed_data[col], errors='coerce')
    trimmed_data.dropna(subset=numerical_cols, inplace=True)

    if trimmed_data.empty:
        st.warning("Processed data is empty. Check your dataset and preprocessing steps.")
    return trimmed_data

# Load the preprocessed data once
data = load_and_preprocess_data()

# Check if data is loaded successfully before proceeding with page rendering
if data.empty:
    st.info("Please ensure 'nigeria_houses_data.csv' is correctly placed and contains valid data.")
    st.stop() # Stop execution if data is empty to prevent further errors


# --- Home page ---
if page == "Home":
    st.markdown("<h1 class='main-title-home'>Nigerian House Price Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.write('#')

    st.markdown("<div class='home-section-container'>", unsafe_allow_html=True)
    st.markdown("<div class='overview-box'>", unsafe_allow_html=True)
    st.subheader("Overview")
    st.write("""
        This dashboard provides an in-depth analysis of house prices in Nigeria.
        You can explore the data, visualize important trends, and see how various features affect house prices.
        Additionally, you can use machine learning models to predict house prices based on input features.
    """)
    st.markdown("</div>", unsafe_allow_html=True) # Close overview-box
    st.markdown("</div>", unsafe_allow_html=True) # Close home-section-container

    st.markdown("<div class='home-section-container'>", unsafe_allow_html=True)
    st.markdown("<div class='pages-box'>", unsafe_allow_html=True)
    st.subheader("Explore the Dashboard:")
    st.markdown("""
        <ul>
            <li>🏠 <b>Home:</b> Overview of the dashboard.</li>
            <li>📊 <b>Dataset:</b> Explore the raw and preprocessed dataset and its features.</li>
            <li>📈 <b>Visualizations:</b> Visualize important trends and insights with interactive charts.</li>
            <li>🧊 <b>3D Visualizations:</b> Explore multi-dimensional relationships in 3D.</li>
            <li>🔍 <b>Filtered Data:</b> Interactively filter the dataset based on various criteria.</li>
            <li>🛠️ <b>Features:</b> Understand feature engineering and its impact.</li>
            <li>🗺️ <b>Data Exploration:</b> Dive deeper into data distributions and relationships.</li>
            <li>📍 <b>Geospatial Analysis:</b> Analyze house prices based on geographical data (aggregated).</li>
            <li>🧠 <b>Model:</b> Train and evaluate machine learning models for house price prediction.</li>
            <li>🔮 <b>Prediction:</b> Predict future house prices based on input parameters.</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Close pages-box
    st.markdown("</div>", unsafe_allow_html=True) # Close home-section-container


# --- Data page ---
elif page == "Dataset":
    st.header("Data Overview")

    st.write('#')
    # Dataset overview
    st.subheader("Dataset")
    st.write("This is the overview of the Nigerian house dataset used for house price prediction.")

    # Display the dataset
    st.dataframe(data)
    st.markdown("---")

    # Display dataset statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    st.markdown("---")

    # Show dataset columns - UPDATED FOR NIGERIAN DATASET
    st.subheader("Column Descriptions")
    st.write("""
        | Column Name              | Description                                       |
        |--------------------------|---------------------------------------------------|
        | `price`                  | Price of the house                                |
        | `bedrooms`               | Number of bedrooms in the house                   |
        | `bathrooms`              | Number of bathrooms in the house                  |
        | `toilets`                | Number of toilets in the house                    |
        | `parking_space`          | Number of parking spaces available                |
        | `title`                  | House type (e.g., Duplex, Detached, Semi-detached)|
        | `town`                   | Town where the house is located                   |
        | `state`                  | State within Nigeria where the house is located   |
        | `total_rooms`            | Engineered feature: sum of bedrooms, bathrooms,   |
        |                          | toilets, and parking_space                        |
    """)
    st.markdown("---")

    # Add interactive filters - UPDATED FOR NIGERIAN DATASET
    st.subheader("Interactive Filters")
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())

    price_filter = st.slider('Select Price Range', min_price, max_price, (min_price, max_price))

    bedrooms_filter_val = st.slider('Select Number of Bedrooms', int(data['bedrooms'].min()), int(data['bedrooms'].max()), (int(data['bedrooms'].min()), int(data['bedrooms'].max())))
    bathrooms_filter_val = st.slider('Select Number of Bathrooms', int(data['bathrooms'].min()), int(data['bathrooms'].max()), (int(data['bathrooms'].min()), int(data['bathrooms'].max())))
    toilets_filter_val = st.slider('Select Number of Toilets', int(data['toilets'].min()), int(data['toilets'].max()), (int(data['toilets'].min()), int(data['toilets'].max())))
    parking_filter_val = st.slider('Select Number of Parking Spaces', int(data['parking_space'].min()), int(data['parking_space'].max()), (int(data['parking_space'].min()), int(data['parking_space'].max())))

    state_filter = st.multiselect('Select State(s)', data['state'].unique(), data['state'].unique())
    town_filter = st.multiselect('Select Town(s)', data['town'].unique(), data['town'].unique())
    title_filter = st.multiselect('Select House Type(s)', data['title'].unique(), data['title'].unique())


    filtered_data = data[
        (data['price'] >= price_filter[0]) & (data['price'] <= price_filter[1]) &
        (data['bedrooms'] >= bedrooms_filter_val[0]) & (data['bedrooms'] <= bedrooms_filter_val[1]) &
        (data['bathrooms'] >= bathrooms_filter_val[0]) & (data['bathrooms'] <= bathrooms_filter_val[1]) &
        (data['toilets'] >= toilets_filter_val[0]) & (data['toilets'] <= toilets_filter_val[1]) &
        (data['parking_space'] >= parking_filter_val[0]) & (data['parking_space'] <= parking_filter_val[1]) &
        (data['state'].isin(state_filter)) &
        (data['town'].isin(town_filter)) &
        (data['title'].isin(title_filter))
    ]
    st.dataframe(filtered_data)


# --- Visualizations page ---
elif page == "Visualizations":
    st.header("House Price Visualizations")
    st.write('#')

    # Distribution of House Prices
    with st.expander("Distribution of House Prices"):
        st.subheader("Distribution of House Prices")
        st.info("This plot shows the distribution of house prices in the dataset. It helps to understand the overall price range and frequency of different price points.")
        fig_hist = px.histogram(data, x='price', nbins=50, title='Distribution of House Prices',
                                color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(bargap=0.2, xaxis_title='Price (₦)', yaxis_title='Count', template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")

    # House Prices by Bedrooms
    with st.expander("House Prices by Number of Bedrooms"):
        st.subheader("House Prices by Number of Bedrooms")
        st.info("This scatter plot shows the relationship between house prices and the number of bedrooms. It helps to visualize the correlation.")
        fig_bedrooms = px.scatter(data, x='bedrooms', y='price', color='state',
                                  title='House Price vs. Bedrooms',
                                  labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price (₦)'},
                                  hover_data=['town', 'title'])
        st.plotly_chart(fig_bedrooms, use_container_width=True)
    st.markdown("---")

    # House Prices by Bathrooms
    with st.expander("House Prices by Number of Bathrooms"):
        st.subheader("House Prices by Number of Bathrooms")
        st.info("This scatter plot shows the relationship between house prices and the number of bathrooms. It helps to visualize the correlation.")
        fig_bathrooms = px.scatter(data, x='bathrooms', y='price', color='state',
                                   title='House Price vs. Bathrooms',
                                   labels={'bathrooms': 'Number of Bathrooms', 'price': 'Price (₦)'},
                                   hover_data=['town', 'title'])
        st.plotly_chart(fig_bathrooms, use_container_width=True)
    st.markdown("---")

    # House Prices by Toilets
    with st.expander("House Prices by Number of Toilets"):
        st.subheader("House Price vs. Number of Toilets")
        st.info("This scatter plot shows the relationship between house prices and the number of toilets. It helps to visualize the correlation.")
        fig_toilets = px.scatter(data, x='toilets', y='price', color='state',
                                 title='House Price vs. Toilets',
                                 labels={'toilets': 'Number of Toilets', 'price': 'Price (₦)'},
                                 hover_data=['town', 'title'])
        st.plotly_chart(fig_toilets, use_container_width=True)
    st.markdown("---")

    # House Prices by Total Rooms (Engineered Feature)
    with st.expander("House Prices by Total Rooms"):
        st.subheader("House Price vs. Total Rooms (Bedrooms + Bathrooms + Toilets + Parking Space)")
        st.info("This scatter plot shows the relationship between house prices and the total number of rooms (an engineered feature).")
        fig_total_rooms = px.scatter(data, x='total_rooms', y='price', color='state',
                                     title='House Price vs. Total Rooms',
                                     labels={'total_rooms': 'Total Rooms', 'price': 'Price (₦)'},
                                     hover_data=['town', 'title'])
        st.plotly_chart(fig_total_rooms, use_container_width=True)
    st.markdown("---")

    # House Type (title) vs. Price (Box Plot)
    with st.expander("House Price by House Type"):
        st.subheader("House Price Distribution by House Type")
        st.info("This box plot shows how house prices vary by the type of house. It helps to identify any patterns or outliers.")
        fig_house_type = px.box(data, x='title', y='price', color='state',
                                title='House Price Distribution by House Type',
                                labels={'title': 'House Type', 'price': 'Price (₦)'})
        st.plotly_chart(fig_house_type, use_container_width=True)
    st.markdown("---")

    # Price by State (Bar Chart)
    with st.expander("Average House Price by State"):
        st.subheader("Average House Price by State")
        st.info("This bar chart shows the average house price across different states in Nigeria.")
        avg_price_state = data.groupby('state')['price'].mean().reset_index().sort_values(by='price', ascending=False)
        fig_state_price = px.bar(avg_price_state, x='state', y='price',
                                 title='Average House Price by State',
                                 labels={'state': 'State', 'price': 'Average Price (₦)'})
        st.plotly_chart(fig_state_price, use_container_width=True)
    st.markdown("---")

    # Price by Town (Top 20 Bar Chart)
    with st.expander("Average House Price by Top 20 Towns"):
        st.subheader("Average House Price by Top 20 Towns")
        st.info("This bar chart highlights the average house prices in the top 20 most expensive towns.")
        avg_price_town = data.groupby('town')['price'].mean().reset_index().sort_values(by='price', ascending=False).head(20)
        fig_town_price = px.bar(avg_price_town, x='town', y='price',
                                title='Average House Price by Top 20 Towns',
                                labels={'town': 'Town', 'price': 'Average Price (₦)'})
        st.plotly_chart(fig_town_price, use_container_width=True)
    st.markdown("---")

    # Correlation Heatmap
    with st.expander("Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        st.info("This heatmap shows the correlation between numerical features in the dataset. Values closer to 1 or -1 indicate stronger correlations.")
        numerical_cols = data.select_dtypes(include=np.number).columns
        corr_matrix = data[numerical_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title='Correlation Heatmap of Numerical Features',
                              color_continuous_scale='Viridis')
        st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("---")


# --- 3D Visualizations Page ---
elif page == "3D Visualizations":
    st.header("3D Visualizations")
    st.write('#')

    if not data.empty:
        # 3D Scatter Plot: Price vs Bedrooms vs Bathrooms
        with st.expander("Price vs Bedrooms vs Bathrooms"):
            st.subheader("3D Scatter Plot: Price vs Bedrooms vs Bathrooms")
            st.info("This 3D scatter plot shows the relationship between house prices, number of bedrooms, and number of bathrooms.")
            st.success("### Components:\n"
                       "X-axis: Number of Bedrooms\n "
                       "Y-axis: Number of Bathrooms\n"
                       "Z-axis: Price of the house\n"
                       "Color: Represents the price (color intensity indicates higher or lower prices).")
            fig_3d_scatter = px.scatter_3d(data, x='bedrooms', y='bathrooms', z='price', color='price',
                                           title='Price vs Bedrooms vs Bathrooms',
                                           color_continuous_scale=px.colors.sequential.Viridis,
                                           hover_data=['town', 'state', 'title', 'total_rooms'])
            fig_3d_scatter.update_layout(scene=dict(
                xaxis_title='Bedrooms',
                yaxis_title='Bathrooms',
                zaxis_title='Price (₦)',
            ),
                width=800,
                height=600,
            )
            st.plotly_chart(fig_3d_scatter, use_container_width=True)
        st.markdown("---")

        # 3D Scatter Plot: Price vs Total Rooms vs Parking Space
        with st.expander("Price vs Total Rooms vs Parking Space"):
            st.subheader("3D Scatter Plot: Price vs Total Rooms vs Parking Space")
            st.info("This 3D scatter plot visualizes the relationship between house prices, total rooms, and parking spaces.")
            st.success("### Components:\n"
                       "X-axis: Total Rooms\n "
                       "Y-axis: Parking Space\n"
                       "Z-axis: Price of the house\n"
                       "Color: Represents the price (color intensity indicates higher or lower prices).")
            fig_3d_scatter_2 = px.scatter_3d(data, x='total_rooms', y='parking_space', z='price', color='price',
                                             title='Price vs Total Rooms vs Parking Space',
                                             color_continuous_scale=px.colors.sequential.Plasma,
                                             hover_data=['town', 'state', 'title', 'bedrooms', 'bathrooms', 'toilets'])
            fig_3d_scatter_2.update_layout(scene=dict(
                xaxis_title='Total Rooms',
                yaxis_title='Parking Space',
                zaxis_title='Price (₦)',
            ),
                width=800,
                height=600,
            )
            st.plotly_chart(fig_3d_scatter_2, use_container_width=True)
        st.markdown("---")

        # 3D Surface Plot: Price as a function of Bedrooms and Bathrooms
        with st.expander("Price as a function of Bedrooms and Bathrooms"):
            st.subheader("3D Surface Plot: Price as a function of Bedrooms and Bathrooms")
            st.info("This 3D surface plot visualizes house prices as a function of bedrooms and bathrooms, showing the continuous variation of price.")
            st.success("### Components:\n"
                       "X-axis: Number of Bedrooms\n"
                       "Y-axis: Number of Bathrooms\n"
                       "Z-axis: Price of the house\n"
                       "Surface: Represents the continuous variation of price across different combinations of bedrooms and bathrooms.")

            # Create a pivot table for the surface plot
            # Ensure unique combinations for pivot table, or aggregate if duplicates
            # Using mean to aggregate prices for same bedroom/bathroom combinations
            # Handle potential empty pivot table if data is too sparse after filtering
            pivot_data = data.pivot_table(index='bedrooms', columns='bathrooms', values='price', aggfunc='mean')
            if not pivot_data.empty:
                fig_3d_surface = go.Figure(
                    data=[go.Surface(x=pivot_data.columns, y=pivot_data.index, z=pivot_data.values,
                                     colorscale='Viridis')])
                fig_3d_surface.update_layout(
                    title='Price as a function of Bedrooms and Bathrooms',
                    scene=dict(
                        xaxis_title='Bedrooms',
                        yaxis_title='Bathrooms',
                        zaxis_title='Price (₦)',
                    ),
                    width=800,
                    height=600,
                )
                st.plotly_chart(fig_3d_surface, use_container_width=True)
            else:
                st.warning("Not enough data to create the 3D surface plot with current filters.")
        st.markdown("---")
    else:
        st.warning("No data available for 3D visualizations. Please check the 'Dataset' page.")


# --- Filtered Data page ---
elif page == "Filtered Data":
    st.header("Filtered Data")
    st.write("## Interactively Filter Your Data")
    st.info("Use the filters below to explore specific subsets of the Nigerian housing dataset.")

    if not data.empty:
        # Filter options - using the same filters as on the Dataset page for consistency
        min_price = int(data['price'].min())
        max_price = int(data['price'].max())

        price_filter = st.slider('Select Price Range (₦)', min_price, max_price, (min_price, max_price))

        bedrooms_filter_val = st.slider('Select Number of Bedrooms', int(data['bedrooms'].min()), int(data['bedrooms'].max()), (int(data['bedrooms'].min()), int(data['bedrooms'].max())))
        bathrooms_filter_val = st.slider('Select Number of Bathrooms', int(data['bathrooms'].min()), int(data['bathrooms'].max()), (int(data['bathrooms'].min()), int(data['bathrooms'].max())))
        toilets_filter_val = st.slider('Select Number of Toilets', int(data['toilets'].min()), int(data['toilets'].max()), (int(data['toilets'].min()), int(data['toilets'].max())))
        parking_filter_val = st.slider('Select Number of Parking Spaces', int(data['parking_space'].min()), int(data['parking_space'].max()), (int(data['parking_space'].min()), int(data['parking_space'].max())))

        state_filter = st.multiselect('Select State(s)', data['state'].unique(), data['state'].unique())
        town_filter = st.multiselect('Select Town(s)', data['town'].unique(), data['town'].unique())
        title_filter = st.multiselect('Select House Type(s)', data['title'].unique(), data['title'].unique())


        filtered_data = data[
            (data['price'] >= price_filter[0]) & (data['price'] <= price_filter[1]) &
            (data['bedrooms'] >= bedrooms_filter_val[0]) & (data['bedrooms'] <= bedrooms_filter_val[1]) &
            (data['bathrooms'] >= bathrooms_filter_val[0]) & (data['bathrooms'] <= bathrooms_filter_val[1]) &
            (data['toilets'] >= toilets_filter_val[0]) & (data['toilets'] <= toilets_filter_val[1]) &
            (data['parking_space'] >= parking_filter_val[0]) & (data['parking_space'] <= parking_filter_val[1]) &
            (data['state'].isin(state_filter)) &
            (data['town'].isin(town_filter)) &
            (data['title'].isin(title_filter))
        ]

        st.write(f"Displaying {len(filtered_data)} filtered data points:")
        st.dataframe(filtered_data)
        st.markdown("---")

        if not filtered_data.empty:
            st.subheader("Filtered Data Statistics")
            st.write(filtered_data.describe())
        else:
            st.warning("No data points match the selected filters.")
    else:
        st.warning("No data available for filtering. Please check the 'Dataset' page.")


# --- Features page ---
elif page == "Features":
    st.header("Feature Engineering Insights")
    st.write("## Displaying Feature Engineering Insights")
    st.info("We created a new feature `total_rooms` by summing `bedrooms`, `bathrooms`, `toilets`, and `parking_space` to gain deeper insights.")

    if not data.empty:
        # Display statistics for the engineered feature
        st.subheader("Statistics for Engineered Feature: `total_rooms`")
        total_rooms_stats = {
            "Metric": ["Mean", "Standard Deviation", "Min", "Max"],
            "Value": [data["total_rooms"].mean(), data["total_rooms"].std(),
                      data["total_rooms"].min(), data["total_rooms"].max()]
        }
        st.write(pd.DataFrame(total_rooms_stats))
        st.markdown("---")

        st.subheader("Impact of `total_rooms` on Price")
        fig_total_rooms_impact = px.scatter(data, x='total_rooms', y='price', color='state',
                                            title='Price vs. Total Rooms',
                                            labels={'total_rooms': 'Total Rooms', 'price': 'Price (₦)'},
                                            hover_data=['town', 'title'])
        st.plotly_chart(fig_total_rooms_impact, use_container_width=True)
        st.markdown("---")
    else:
        st.warning("No data available for feature insights. Please check the 'Dataset' page.")


# --- Data Exploration page ---
elif page == "Data Exploration":
    st.header("Data Exploration")
    st.info("Explore the dataset with interactive visualizations to understand distributions and relationships between key features.")

    if not data.empty:
        # Scatter Matrix of Key Numerical Features
        st.subheader("Scatter Matrix of Key Features")
        explore_cols = ['price', 'bedrooms', 'bathrooms', 'toilets', 'total_rooms']
        fig_explore = px.scatter_matrix(data[explore_cols], dimensions=explore_cols,
                                        title='Scatter Matrix of Key Housing Features',
                                        color='price',
                                        color_continuous_scale=px.colors.sequential.Plasma)
        fig_explore.update_layout(height=800, width=800)
        st.plotly_chart(fig_explore, use_container_width=True)
        st.markdown("---")

        st.write("### Model Comparison (Placeholder)")
        st.info("This section will compare different models' performances. The actual model training and evaluation is done on the 'Model' page.")
        st.write("Please navigate to the 'Model' page to train and evaluate machine learning models.")
        st.markdown("---")
    else:
        st.warning("No data available for data exploration. Please check the 'Dataset' page.")


# --- Geospatial Analysis page ---
elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    st.write("## Analyzing House Prices by Location in Nigeria")
    st.info("This section provides insights into house price variations across different states and towns in Nigeria. Note: Precise latitude/longitude data is not available in this dataset for point-level mapping, so we focus on aggregated views.")

    if not data.empty:
        # Average Price by State
        st.subheader("Average House Price by State")
        avg_price_state = data.groupby('state')['price'].mean().reset_index().sort_values(by='price', ascending=False)
        fig_state_price_geo = px.bar(avg_price_state, x='state', y='price',
                                     title='Average House Price by State (₦)',
                                     labels={'state': 'State', 'price': 'Average Price (₦)'},
                                     color='price', color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_state_price_geo, use_container_width=True)
        st.markdown("---")

        # Average Price by Top 10 Towns
        st.subheader("Average House Price by Top 10 Towns")
        avg_price_town_geo = data.groupby('town')['price'].mean().reset_index().sort_values(by='price', ascending=False).head(10)
        # FIX IS ON THE NEXT LINE: Changed 'avg_town_price_geo' to 'avg_price_town_geo'
        fig_town_price_geo = px.bar(avg_price_town_geo, x='town', y='price', # THIS LINE IS CORRECTED
                                    title='Average House Price by Top 10 Towns (₦)',
                                    labels={'town': 'Town', 'price': 'Average Price (₦)'},
                                    color='price', color_continuous_scale=px.colors.colors.sequential.Plasma)
        st.plotly_chart(fig_town_price_geo, use_container_width=True)
        st.markdown("---")
    else:
        st.warning("No data available for geospatial analysis. Please check the 'Dataset' page.")


# --- Model Page ---
elif page == "Model":
    st.header("Train and Evaluate Machine Learning Models")
    st.write("## Select a model and train it to predict house prices.")

    if not data.empty:
        # Prepare data for modeling
        modeling_data = data[['bedrooms', 'bathrooms', 'toilets', 'total_rooms', 'price']].copy()

        features_for_model = ['bedrooms', 'bathrooms', 'toilets', 'total_rooms']
        target = 'price'

        X = modeling_data[features_for_model]
        y = modeling_data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Ridge Regression", "Lasso Regression",
                                                             "Decision Tree", "Random Forest", "Gradient Boosting Regressor",
                                                             "K-Nearest Neighbors", "Support Vector Regressor"])

        model = None

        if st.button(f"Train {model_choice} Model"):
            with st.spinner(f"Training the {model_choice} model, please wait..."):
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Ridge Regression":
                    alpha = st.sidebar.slider("Alpha", min_value=0.01, max_value=10.0, value=1.0)
                    model = Ridge(alpha=alpha)
                elif model_choice == "Lasso Regression":
                    alpha = st.sidebar.slider("Alpha", min_value=0.01, max_value=10.0, value=1.0)
                    model = Lasso(alpha=alpha)
                elif model_choice == "Decision Tree":
                    max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=20, value=5)
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif model_choice == "Random Forest":
                    n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=200, value=100)
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                elif model_choice == "Gradient Boosting Regressor":
                    n_estimators = st.sidebar.slider("n_estimators", min_value=50, max_value=500, value=200)
                    learning_rate = st.sidebar.slider("learning_rate", min_value=0.01, max_value=0.5, value=0.1)
                    max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=10, value=3)
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                                      max_depth=max_depth, random_state=42)
                elif model_choice == "K-Nearest Neighbors":
                    n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=20, value=5)
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                elif model_choice == "Support Vector Regressor":
                    C = st.sidebar.slider("C (Regularization)", min_value=0.1, max_value=10.0, value=1.0)
                    epsilon = st.sidebar.slider("Epsilon (SVR tolerance)", min_value=0.01, max_value=1.0, value=0.1)
                    model = SVR(C=C, epsilon=epsilon)

                if model:
                    model.fit(X_train, y_train)

                    # Save the trained model
                    joblib.dump(model, 'house_price_model.pkl')
                    st.session_state['trained_model'] = model # Store in session state for immediate use

                    # Predict and evaluate the model
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Display evaluation metrics
                    st.write(f"### Model: {model_choice}")
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                    st.metric("Mean Absolute Error", f"{mae:.2f}")
                    st.metric("R-squared", f"{r2:.4f}")

                    st.write("---")
                    st.write("#")

                    # Visualization using Plotly: Actual vs. Predicted House Prices
                    st.write("### Actual vs. Predicted House Prices")
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Actual'],
                        y=results_df['Predicted'],
                        mode='markers',
                        name='Predicted vs. Actual',
                        marker=dict(color='yellow', size=6, line=dict(width=1))
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['Actual'],
                        y=results_df['Actual'],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))

                    fig.update_layout(
                        title=f"{model_choice} - Actual vs Predicted Prices",
                        xaxis_title="Actual Prices (₦)",
                        yaxis_title="Predicted Prices (₦)",
                        showlegend=True,
                        legend=dict(x=0, y=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Model could not be initialized. Please check parameters.")
        else:
            st.info("Click 'Train Model' to train the selected model.")
    else:
        st.warning("No data available for model training. Please check the 'Dataset' page.")


# --- Prediction Page ---
elif page == "Prediction":
    st.header("Predict House Prices")
    st.write("Enter the features of the house to get a predicted price.")

    # Load the trained model (if available)
    model_file_path = 'house_price_model.pkl'
    model = None
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        st.success("Model loaded from session state!")
    elif os.path.exists(model_file_path):
        try:
            model = joblib.load(model_file_path)
            st.success("Model loaded from file!")
        except Exception as e:
            st.error(f"Error loading model from file: {e}. Please train a model on the 'Model' page first.")
    else:
        st.warning("No trained model found. Please train a model on the 'Model' page first.")

    st.markdown("---")
    st.write("### Input Features for Prediction")

    if not data.empty:
        bedrooms = st.number_input("Number of Bedrooms", min_value=int(data['bedrooms'].min()), max_value=int(data['bedrooms'].max()), value=int(data['bedrooms'].mean()))
        bathrooms = st.number_input("Number of Bathrooms", min_value=int(data['bathrooms'].min()), max_value=int(data['bathrooms'].max()), value=int(data['bathrooms'].mean()))
        toilets = st.number_input("Number of Toilets", min_value=int(data['toilets'].min()), max_value=int(data['toilets'].max()), value=int(data['toilets'].mean()))
        parking_space = st.number_input("Number of Parking Spaces", min_value=int(data['parking_space'].min()), max_value=int(data['parking_space'].max()), value=int(data['parking_space'].mean()))

        total_rooms = bedrooms + bathrooms + toilets + parking_space
        st.info(f"Calculated Total Rooms: {total_rooms}")

        st.markdown("---")
        st.write("#")
        st.markdown("### Input Summary")
        col1, col2 = st.columns(2)
        col1.metric("Bedrooms", bedrooms)
        col1.metric("Bathrooms", bathrooms)
        col1.metric("Toilets", toilets)
        col2.metric("Parking Spaces", parking_space)
        col2.metric("Total Rooms", total_rooms)

        input_df = pd.DataFrame([[bedrooms, bathrooms, toilets, total_rooms]],
                                columns=['bedrooms', 'bathrooms', 'toilets', 'total_rooms'])

        if st.button("Predict Price"):
            if model:
                st.info("Predicting the house price based on your input...")
                try:
                    prediction = model.predict(input_df)
                    st.success(f"The predicted price of the house is approximately ₦{prediction[0]:,.2f}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.warning("Please ensure the model is trained correctly and inputs are valid.")
            else:
                st.warning("A model needs to be trained on the 'Model' page before making predictions.")
    else:
        st.warning("No data available to define prediction input ranges. Please check the 'Dataset' page.")


# Footer
st.sidebar.markdown(f"""
---
**Developed by ADESHIGBIN OYINDAMOLA AJARAT FTP/CSC/24/0090164**
""")
