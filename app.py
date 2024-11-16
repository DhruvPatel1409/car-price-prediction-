import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.pipeline import Pipeline

# Load the trained model
Lmodel = pickle.load(open('model.pkl', 'rb'))

# Define the prediction function
def predict_price(input_data):
    return Lmodel.predict(input_data)

# Function to display scatter plot
def display_scatter_plot(df, x, y, color=None, size=None, animation_frame=None):
    scatter_plot = px.scatter(df, x=x, y=y, color=color, size=size, animation_frame=animation_frame,
                              title=f'{x} vs {y}', labels={x: x, y: y})
    st.plotly_chart(scatter_plot)

# Function to display bar chart
def display_bar_chart(df, x, y, color=None, animation_frame=None):
    bar_chart = px.bar(df, x=x, y=y, color=color, animation_frame=animation_frame,
                      title=f'{y} by {x}', labels={x: x, y: y})
    st.plotly_chart(bar_chart)

# Function to display box plot
def display_box_plot(df, x, y, color=None, animation_frame=None):
    box_plot = px.box(df, x=x, y=y, color=color, animation_frame=animation_frame,
                      title=f'{y} by {x}', labels={x: x, y: y})
    st.plotly_chart(box_plot)

# Function to display pie chart
def display_pie_chart(df, names, values):
    pie_chart = px.pie(df, names=names, values=values, title='Distribution of Cars by Company')
    st.plotly_chart(pie_chart)

# Create the Streamlit web app
def main():
    st.title('Car Price Prediction App')

    # Navigation
    with st.sidebar:
        selected = option_menu("Navigate", ['Home', 'Prediction', 'Visualizations'],icons=['bi-house','bi-currency-rupee','bi-crosshair2'])

    if selected == 'Home':
        st.markdown(
            "This app predicts the price of a car based on various features. Navigate to the 'Prediction' page to make predictions "
            "or the 'Visualizations' page to explore data visualizations."
        )
        st.image('car_image.jpg', use_column_width=True)

    elif selected == 'Prediction':

        name = st.text_input('Car Name')
        company = st.text_input('Company')
        year = st.number_input('Year', min_value=1900, max_value=2024, step=1)
        kms_driven = st.number_input('Kilometers Driven', min_value=0, step=1000)
        fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'LPG'])

        # Predict button
        if st.button('Predict'):
            input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

            prediction = predict_price(input_df)
            st.success(f'Predicted Price: {prediction[0]}')

    elif selected == 'Visualizations':
    
        df = pd.read_csv('cars.csv')
        
        visualization_type = st.sidebar.selectbox('Select Visualization', ['Scatter Plot', 'Bar Chart', 'Box Plot', 'Pie Chart'])

        if visualization_type == 'Scatter Plot':
            st.subheader('Scatter Plot')
            x = st.selectbox('X-axis', df.columns)
            y = st.selectbox('Y-axis', df.columns)
            color = st.selectbox('Color', [None] + df.columns.tolist())
            animation_frame = st.selectbox('Animation Frame', [None] + df.columns.tolist())
            display_scatter_plot(df, x, y, color, animation_frame)

        elif visualization_type == 'Bar Chart':
            st.subheader('Bar Chart')
            x = st.selectbox('X-axis', df.columns)
            y = st.selectbox('Y-axis', df.columns)
            color = st.selectbox('Color', [None] + df.columns.tolist())
            animation_frame = st.selectbox('Animation Frame', [None] + df.columns.tolist())
            display_bar_chart(df, x, y, color, animation_frame)

        elif visualization_type == 'Box Plot':
            st.subheader('Box Plot')
            x = st.selectbox('X-axis', df.columns)
            y = st.selectbox('Y-axis', df.columns)
            color = st.selectbox('Color', [None] + df.columns.tolist())
            animation_frame = st.selectbox('Animation Frame', [None] + df.columns.tolist())
            display_box_plot(df, x, y, color, animation_frame)

        elif visualization_type == 'Pie Chart':
            st.subheader('Pie Chart')
            names = st.selectbox('Names', df.columns)
            values = st.selectbox('Values', ['company'])
            display_pie_chart(df, names, values)

if __name__ == '__main__':
    main()
