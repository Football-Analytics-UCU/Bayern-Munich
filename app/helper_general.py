import streamlit as st
import pandas as pd


def get_binary_chart_wrapper(data, title, *args, **kwargs):

    return get_binary_chart(data[title]['Netherlands'], data[title]['Ukraine'], title, *args, **kwargs)


def get_binary_chart(nth, ukr, title, format="%"):
    is_zero = not (nth or ukr)

    if format == "%" and not is_zero:
        total_sum = nth + ukr
        nth /= total_sum
        ukr /= total_sum

    st.header(title)

    if is_zero:
        format = 'Z'
        ax = pd.DataFrame({'a': [1, 0.004, 1]}).T.plot.barh(stacked=True, figsize=(10, 1), color=['lightgray', 'white', 'lightgray'])
    else:
        ax = pd.DataFrame({'a': [nth, ukr]}).T.plot.barh(stacked=True, figsize=(10, 1), color=['orange', 'dodgerblue'])
    ax.legend().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for c, lb in zip(ax.containers, ['Netherlands', '', 'Ukraine'] if is_zero else ['Netherlands', 'Ukraine']):
        if c.datavalues and lb:
            ax.bar_label(
                c,
                label_type='center',
                labels=[{'%': f'{cc:.1%} - {lb}', 'N': f'{cc:.0f} - {lb}', 'F': f'{cc:.2f} - {lb}', 'Z': f'0 - {lb}'}[format] for cc in c.datavalues],
                color='white'
            )

    fig = ax.get_figure()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    return fig


def get_data(df_events):
    res_df = df_events.groupby('team').agg({'possession': "sum"}).rename(columns={'possession': 'Possession'})

    return res_df



def create_general_tab(df_events):

    data = get_data(df_events)

    st.title("Performance")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Possession", format='%'))
        with col2:
            st.pyplot(get_binary_chart(688, 424, "Passes attempted", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(617, 360, "Passes completed", format='N'))
        with col2:
            st.pyplot(get_binary_chart(97.82, 99.39, "Distance covered (km)", format='F'))

    st.title("Attacking")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(3, 2, "Goals", format='N'))
        with col2:
            st.pyplot(get_binary_chart(15, 8, "Total attempts", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(7, 5, "Attempts on target", format='N'))
        with col2:
            st.pyplot(get_binary_chart(4, 2, "Attempts off target", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(4, 1, "Blocked", format='N'))
        with col2:
            st.pyplot(get_binary_chart(0, 0, "WoodWork", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(5, 1, "Corners taken", format='N'))
        with col2:
            st.pyplot(get_binary_chart(2, 1, "Offsides", format='N'))

    st.title("Defending")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(43, 43, "Balls recovered", format='N'))
        with col2:
            st.pyplot(get_binary_chart(7, 12, "Tackles", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(1, 4, "Blocks", format='N'))
        with col2:
            st.pyplot(get_binary_chart(4, 15, "Clearances completed", format='N'))

    st.title("Disciplinary")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(0, 1, "Yellow cards", format='N'))
        with col2:
            st.pyplot(get_binary_chart(0, 0, "Red cards", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart(8, 8, "Fouls committed", format='N'))
        #with col2:
        #    st.pyplot(get_binary_chart(4, 15, "Clearances completed", format='N'))