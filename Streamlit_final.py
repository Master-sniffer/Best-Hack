import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Load parquet file
@st.cache(allow_output_mutation=True)
def load_parquet(file):
    return pq.read_table(file).to_pandas()

file = st.file_uploader("Upload your parquet file", type="parquet")
if file is not None:
    df = load_parquet(file)
    matplotlib.use('TkAgg')
    # Visualize NaN values
    st.subheader("NaN values visualization")
    fig = px.imshow(pd.isna(df), width=800, title="NaN values in the dataset")
    fig.update_layout(coloraxis=dict(colorscale='Blues'), title_font=dict(size=18))
    st.plotly_chart(fig)

    pd.crosstab(df.vidsobst,df.fr_group,margins=True).style.background_gradient(cmap='summer_r')
    st.pyplot(sns.displot(data=df, x='vidsobst'))

    column_name = 'prev_distance'
    figure = plt.figure(figsize=(10, 5))
    plt.hist(df[column_name])
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_name} with NaN values')
    st.pyplot(plt)
    #plt.show()

    sns.displot(df['prev_date_arrival'], kde=False)
    st.pyplot(plt)
    #plt.show()
