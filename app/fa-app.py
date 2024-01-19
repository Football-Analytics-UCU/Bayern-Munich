import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def plot_heatmap(df):
	fig, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(df, annot=True, linewidths=.5, ax=ax)
	st.pyplot(fig, use_container_width=True)

st.set_page_config(page_title='bayern-project', layout="wide")

path_data_events = "events.csv"
df_events = pd.read_csv(path_data_events)

_, col01, _ = st.columns((1, 1, 1))
with col01:
    st.title('Netherlands 3-2 Ukraine')

tab1, tab2, tab3 = st.tabs(["General", "Pass", "Goal"])

with tab1:
    st.write(df_events.head(10))

with tab2:

	df_pass = df_events[df_events['team']=='Ukraine'].pivot_table(
	    values='id', 
	    index='player', 
	    columns='pass_recipient', 
	    aggfunc='count'
	)

	_, col02 = st.columns((1, 1))
	with col02:
		plot_heatmap(df_pass)

with tab3:
    pass