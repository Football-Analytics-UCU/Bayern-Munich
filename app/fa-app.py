import streamlit as st
import pandas as pd
from helper_passes import (get_passes_data, filter_passes_data, plot_pass_map, plot_pass_heatmap)
import rp_plot_functions

st.set_page_config(page_title='bayern-project', layout="wide")

# path_data_events = "app/data/events.csv"
# df_events = pd.read_csv(path_data_events)

path_data_events = "app/data/events.pkl"
df_events = pd.read_pickle(path_data_events)


_, col01, _ = st.columns((1, 1, 1))
with col01:
    st.title('Netherlands 3-2 Ukraine')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["General", "Passes", "Goals", "Corners", "Throw-in", "About"])


with tab1:
    st.write(df_events.tail(10))


with tab2:
    
    _, col020, _ = st.columns((1, 1, 1))
    with col020:
        st.markdown('### Passing Networks')
        selected_min_value = st.slider('Select Minimum Number Passes btw Players:', min_value=1, max_value=5, value=2, step=1, key=21)

    config_color_dict = {0: {'cmap': 'Oranges', 'color': 'orange'}, 1: {'cmap': 'Blues', 'color': 'dodgerblue'}}
    minute_init = [{'minute': -1, 'period': 1, 'timestamp': '00:00:00.000', 'type': 'Game Start'}]
    minute_last = [{'minute': 95, 'period': 2, 'timestamp': '90:00:00.000', 'type': 'Game Over'}]
    break_events = ['Substitution', 'Red Card']

    breaks_dict = {}
    for team in df_events['team'].unique():
        df_events_break = df_events[(df_events["type"].isin(break_events))&
                                    (df_events["team"]==team)][['team', 'period', 'minute', 'type', 'timestamp']]
        breaks_list = df_events_break.groupby(['minute', 'period', 'type'])['timestamp'].max().reset_index().to_dict('records')
        breaks_dict[team] = minute_init + breaks_list + minute_last

    config_dict = {}
    for idx_team, team in enumerate(breaks_dict.keys()):
        idx_list = []
        for i in range(len(breaks_dict[team])-1):
            period_start = breaks_dict[team][i]
            period_end = breaks_dict[team][i+1]

            df_player_pass, df_player_location = get_passes_data(df_events, team, period_start['minute'], period_end['minute'])
            df_player_pass, df_player_location = filter_passes_data(df_player_pass, df_player_location, min_pass_count=selected_min_value)

            idx_list.append({
                'idx': i,
                'team': team,
                'data_pass': df_player_pass,
                'data_location': df_player_location,
                'metric': {
                    'max_passes': df_player_location['passes'].max() if not df_player_pass.empty else 0,
                    'min_passes': df_player_location['passes'].min() if not df_player_pass.empty else 0,
                    'max_passes_direction': df_player_pass['passes'].max() if not df_player_pass.empty else 0,
                    'min_passes_direction': df_player_pass['passes'].min() if not df_player_pass.empty else 0,
                    },
                'period_start': period_start, 
                'period_end': period_end,
                'color': config_color_dict[idx_team]['color'],
                'cmap': config_color_dict[idx_team]['cmap'],
            })
        config_dict[team] = idx_list

    for k in config_dict.keys():
        # st.write(k, len(config_dict[k]))
        # st.write(config_dict[k])

        num_cols = len(config_dict[k])
        cols = st.columns(num_cols)

        for i in range(num_cols):
            col = cols[i]
            with col:
                plot_pass_map(
                    config_dict[k][i]['data_pass'],
                    config_dict[k][i]['data_location'],
                    team_name=config_dict[k][i]['team'],
                    metric=config_dict[k][i]['metric'],
                    period_start=config_dict[k][i]['period_start'],
                    period_end=config_dict[k][i]['period_end'],
                    color=config_dict[k][i]['color'],
                    cmap_name=config_dict[k][i]['cmap']
                )

    df_pass = df_events[df_events['team']=='Ukraine'].pivot_table(
        values='id', 
        index='player', 
        columns='pass_recipient', 
        aggfunc='count'
    )

    _, col02 = st.columns((1, 1))
    with col02:
        plot_pass_heatmap(df_pass)


with tab3:
    pass

with tab4:
    st.title('Corner events in our data')
    #1
    st.text('The player who get the ball out of bounds: Stefan de Vrij. Method: Clearance')
    IDS = ['bf6c9261-840b-4d8f-b7bd-9daa23b5b457']
    FF = ['corner']
    rp_plot_functions.make_graph(IDS, FF, 'First corner pass')
    #1 result
    IDS = ['80110e08-410a-4928-9601-3d2f0694589f'] 
    FF = ['freeze']
    rp_plot_functions.make_graph(IDS, FF, 'Result: Off target shot')
    #2
    st.text('The player who get the ball out of bounds: Heorhii Bushchan. Method: Goalkeeper saves')
    IDS = ['49f48dc3-8bf0-41c8-af49-ab4ec4b164ee']
    FF = ['corner']
    rp_plot_functions.make_graph(IDS, FF, 'Second corner pass')
    #3
    st.text('The player who get the ball out of bounds: Mykola Matviyenko. Method: Clearance')
    IDS = ['022d0dc4-9779-4224-9533-ef690e560907']
    FF = ['corner']
    rp_plot_functions.make_graph(IDS, FF, 'Third corner pass')
    #3 result
    IDS = ['9f9d09f9-c091-4c58-b7f0-873df2649cbd', '022d0dc4-9779-4224-9533-ef690e560907']
    FF = ['freeze']
    rp_plot_functions.make_graph(IDS, FF, 'Result: Off target shot')
    #4
    st.text('The player who get the ball out of bounds: Illia Zabarnyi. Method: Clearance')
    IDS = ['cfafc900-4c4d-49d5-b11c-8a3e0390eeca', '7545d8df-3938-4662-b1e0-cf45bb49b338']
    FF = ['solo', 'corner']
    rp_plot_functions.make_graph(IDS, FF, 'Fourth corner pass')
    #5
    st.text('The player who get the ball out of bounds: Oleksandr Karavaev. Method: Interception')
    IDS = ['9b62f73f-60d0-4cf2-bef1-7aeee121f6af', 'ee9667f9-4f49-4c61-890a-91685aef779b']
    FF = ['solo', 'corner']
    rp_plot_functions.make_graph(IDS, FF, 'Fifth corner pass')
    #6
    st.text('The player who get the ball out of bounds: Vitalii Mykolenko. Method: Duel')
    IDS = ['39755e7b-b68e-46f4-9fd9-07ac6b41ef2c']
    FF = ['corner']
    rp_plot_functions.make_graph(IDS, FF, 'Sixth corner pass')
    # Mean corner
    st.title('Mean num. of corner events in groups')
    values = [12.0, 6.5, 8.666666666666666, 9.166666666666666, 9.666666666666666, 5.833333333333333, 8.375, 14.25, 9.0, 8.0]
    v_names = ['A', 'B', 'C', 'D', 'E', 'F', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final']
    rp_plot_functions.make_chart_bart(values, v_names, 2)

with tab5:
    pass

with tab6:
    st.markdown('### Team')
    st.markdown('### References')
    st.markdown("""
        - [Match Video](https://www.tokyvideo.com/video/netherlands-3-2-ukraine-full-match-euro-2020-group-stage-13-6-2021)
        - [Match Report: FBREF](https://fbref.com/en/matches/0e9919a5/Netherlands-Ukraine-June-13-2021-European-Championship)
        """)
