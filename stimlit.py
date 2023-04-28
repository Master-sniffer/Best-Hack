import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read Parquet file
@st.cache
def load_data(file):
    data = pd.read_csv(file, engine='pyarrow')
    return data

# Function to display charts and visualizations
def show_data(data):
    df=data
    st.header("Numerical Data Visualization")
    st.line_chart(data)
    st.bar_chart(data)
    st.area_chart(data)
    st.altair_chart(data)
    #st.table(data)

    st.write("### Data Plot")
    sns.heatmap(data.isna())

    for col in data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data[col], kde=True, ax=ax)
        sns.boxplot(data[col], ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)


    def plot_nan_counts(data):
        nan_counts = data.isnull().sum()
        x = nan_counts.index.values
        y = nan_counts.values
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_xticklabels(x, rotation=90)
        ax.set_title("Number of NaNs by Column")
        ax.set_xlabel("Column Name")
        ax.set_ylabel("Number of NaNs")
        return fig

    st.subheader("Number of NaNs by Column")
    fig = plot_nan_counts(data)
    st.pyplot(fig)

    st.subheader("Histograms")
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            fig, ax = plt.subplots()
            ax.hist(data[col].values)
            ax.set_title(col)
            st.pyplot(fig)

    # Plot histograms of each numerical column
    for column in df.select_dtypes(include=[np.number]).columns.tolist():
        plt.hist(df[column], bins=20)
        st.pyplot()

    # Plot boxplots of each numerical column
    for column in df.select_dtypes(include=[np.number]).columns.tolist():
        plt.boxplot(df[column])
        st.pyplot()

    # Show the correlation matrix
    corr_matrix = df.corr()
    st.write(corr_matrix)

    # Heatmap of the correlation matrix
    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix.to_numpy())
    st.pyplot()

    # Scatter plot of two columns with highest correlation
    highest_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)[0]
    x_column, y_column = np.where(corr_matrix == highest_corr)[0]
    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:,x_column], df.iloc[:,y_column])
    st.pyplot()


    def highlight_nans(s):
        return ['color: red' if np.isnan(v) else '' for v in s]

    # Function to perform analysis and display results
    def perform_analysis(df):
        st.write("### Analysis Results")

    # Mean value of each column
    means = df.mean()
    st.write("#### Mean Values")
    st.write(means)

    # Median value of each column
    medians = df.median()
    st.write("#### Median Values")
    st.write(medians)

    # Max value of each column
    maxes = df.max()
    st.write("#### Max Values")
    st.write(maxes)

    # Min value of each column
    mins = df.min()
    st.write("#### Min Values")
    st.write(mins)


    # Sidebar for selecting number of rows to display
    num_rows = st.sidebar.number_input("Number of rows to display:", min_value=1, max_value=len(df), value=10)

    # Display table of data with highlighted NaNs
    st.write("### Data Table")
    st.write(df.iloc[:num_rows].style.apply(highlight_nans))

    # Graphs
    st.write("### Graphs")
    st.line_chart(df)
    st.bar_chart(df)

    # Perform analysis
    perform_analysis(df)


# Main function to run the app
def main():
    st.title("Data Visualization with Streamlit")

# Add file uploader widget to sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    get_answer = st.button("Get answer")
    if (get_answer):
        st.text("Predicting...")
        # TODO Append predicting for BESTHACK

    if uploaded_file is not None:
    # Load data using the 'load_data' function
        data = load_data(uploaded_file)
        # Show data using the 'show_data' function
        show_data(data)

if __name__ == '__main__':
    main()