import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd

from helper_passes import (get_passes_data, filter_passes_data, get_passes_player_data, create_pass_heatmap, plot_hulls, create_substitution_data, plot_pass_map, plot_pass_heatmap)

st.set_page_config(page_title='bayern-project', layout="wide")

path_data_events = "app/data/events.pkl"
path_data_lineups = "app/data/lineups.pkl"
df_events = pd.read_pickle(path_data_events)
df_lineups = pd.read_pickle(path_data_lineups)

df_substitution = df_events[df_events['substitution_replacement'].notna()]\
    [['player', 'substitution_replacement', 'minute', 'substitution_outcome']]

ev = df_events[
    (df_events['player'].notna())&
    (df_events['type'] == 'Pass')
    ][['minute', 'second', 'team', 'location', 'type', 'player', 'pass_end_location', 'pass_outcome']].copy()

ev_ukr = ev[(ev['team'] == 'Ukraine')].copy().reset_index(drop=True)
ev_nth = ev[(ev['team'] != 'Ukraine')].copy().reset_index(drop=True)

lineups_ukr = df_lineups[df_lineups['country'] == 'Ukraine'].copy()
lineups_nth = df_lineups[df_lineups['country'] == 'Netherlands'].copy()

config_color_dict = {0: {'cmap': 'Oranges', 'color': 'orange'}, 1: {'cmap': 'Blues', 'color': 'dodgerblue'}}
minute_init = [{'minute': -1, 'period': 1, 'timestamp': '00:00:00.000', 'type': 'Game Start'}]
minute_last = [{'minute': 95, 'period': 2, 'timestamp': '90:00:00.000', 'type': 'Game Over'}]

_, col01, _ = st.columns((1, 1, 1))
with col01:
    st.title(':orange[Netherlands] 3-2 :blue[Ukraine]')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "General", "Passes", "Player Passing Network", "Team Passing Network", "Goals", "Corners", "Throw-in", "About"])


with tab1:
    st.write(df_lineups.tail(10))
    st.write(df_events.tail(10))

with tab2:
    df_pass = df_events[df_events['team']=='Ukraine'].pivot_table(
        values='id', 
        index='player', 
        columns='pass_recipient', 
        aggfunc='count'
    )

    _, col201, _ = st.columns((0.1, 1, 0.1))
    with col201:
        st.markdown("### :orange[Netherlands] players Convex Hulls")
        plot_hulls(ev_nth, lineups_nth, main_color='orange')
        st.markdown("### :blue[Ukraine] players Convex Hulls")
        plot_hulls(ev_ukr, lineups_ukr, main_color='dodgerblue')

        st.markdown("### Pass Clustering")
        sel_clusters = st.slider('Number of clusters:', min_value=10, max_value=150, value=90, step=10)
        st.markdown("### :orange[Netherlands]")
        create_pass_heatmap(ev_nth, sel_clusters)
        st.markdown("### :blue[Ukraine]")
        create_pass_heatmap(ev_ukr, sel_clusters)


with tab3:
    _, col030, _ = st.columns((1, 1, 1))
    with col030:
        selected_min_value = st.slider('Select Minimum Number Passes btw Players:', min_value=1, max_value=5, value=1, step=1, key=31)
        selected_num_cols_tab3 = st.slider('Select Nunber Columns on screen:', min_value=1, max_value=5, value=4, step=1, key=32)

    config_dict_player = {}
    for idx_team, team in enumerate(df_events['team'].unique()):

        df_events_team = df_events[
            (df_events['type']=='Pass')&
            (df_events['pass_outcome'].isna())&
            (df_events['team']==team)
        ].copy().reset_index(drop=True)

        df_substitution_team = create_substitution_data(df_events_team, df_substitution)

        team_list = []
        for i, row in df_substitution_team.iterrows():
            
            df_events_player = df_events_team[
                (df_events_team['player']==row['player_x'])
            ].copy().reset_index(drop=True)

            substitution_player_dict = df_substitution_team[
                (df_substitution_team['player_x']==row['player_x'])
            ].copy().to_dict('records')

            df_player_pass, df_player_location = get_passes_player_data(df_events_player)
            df_player_pass, df_player_location = filter_passes_data(df_player_pass, df_player_location, min_pass_count=selected_min_value)

            team_list.append({
                'team': team,
                'player': row['player_x'],
                'substitution': substitution_player_dict[0],
                'data_pass': df_player_pass,
                'data_location': df_player_location,
                'metric': {
                    'max_passes': df_player_location['passes'].max() if not df_player_pass.empty else 0,
                    'min_passes': df_player_location['passes'].min() if not df_player_pass.empty else 0,
                    'max_passes_direction': df_player_pass['passes'].max() if not df_player_pass.empty else 0,
                    'min_passes_direction': df_player_pass['passes'].min() if not df_player_pass.empty else 0,
                    },
                'color': config_color_dict[idx_team]['color'],
                'cmap': config_color_dict[idx_team]['cmap'],
            })
        config_dict_player[team] = team_list

    for k in config_dict_player.keys():
        # st.write(k, len(config_dict_player[k]))
        # st.write(config_dict_player[k])

        cols_all = len(config_dict_player[k])
        cols = st.columns(selected_num_cols_tab3)

        for i in range(cols_all):
            col = cols[i%selected_num_cols_tab3]

            sub_start = config_dict_player[k][i]['substitution']['start']
            sub_end = config_dict_player[k][i]['substitution']['end']
            sub_in = config_dict_player[k][i]['substitution']['substitution_replacement_x']
            sub_out = config_dict_player[k][i]['substitution']['player_y']
            sub_reason = config_dict_player[k][i]['substitution']['substitution_outcome']

            if sub_start=='Game Start' and sub_end=='Game Over':
                sub_title = "Played Full Time"
            elif sub_start=='Game Start' and sub_end!='Game Over':
                sub_title = f"Substitited by {sub_in} on {sub_end} min. Reason: {sub_reason}."
            elif sub_start!='Game Start' and sub_end=='Game Over':
                sub_title = f"Substituted instead of {sub_out} on {sub_start} min."
            else:
                sub_title = f"Substituted instead of {sub_out} on {sub_start} min.\nAnd substitited by {sub_in} on {sub_end} min. Reason: {sub_reason}."
            
            with col:
                plot_pass_map(
                    config_dict_player[k][i]['data_pass'],
                    config_dict_player[k][i]['data_location'],
                    title=config_dict_player[k][i]['player'],
                    sub_title=sub_title,
                    metric=config_dict_player[k][i]['metric'],
                    color=config_dict_player[k][i]['color'],
                    cmap_name=config_dict_player[k][i]['cmap']
                )

with tab4:
    _, col020, _ = st.columns((1, 1, 1))
    with col020:
        selected_break_events = st.multiselect('Select Break Events:', ['Substitution', 'Red Card'], ['Substitution'], key=41)
        selected_min_value = st.slider('Select Minimum Number Passes btw Players:', min_value=1, max_value=5, value=2, step=1, key=42)
        selected_num_cols_tab4 = st.slider('Select Nunber Columns on screen:', min_value=1, max_value=5, value=3, step=1, key=43)

    breaks_dict = {}
    for team in df_events['team'].unique():
        df_events_break = df_events[(df_events["type"].isin(selected_break_events))&
                                    (df_events["team"]==team)][['team', 'period', 'minute', 'type', 'timestamp']]
        breaks_list = df_events_break.groupby(['minute', 'period', 'type'])['timestamp'].max().reset_index().to_dict('records')
        breaks_dict[team] = minute_init + breaks_list + minute_last

    config_dict = {}
    for idx_team, team in enumerate(breaks_dict.keys()):
        
        df_events_team = df_events[
            (df_events['type']=='Pass')&
            (df_events['pass_outcome'].isna())&
            (df_events['team']==team)
        ].copy().reset_index(drop=True)

        team_list = []
        for i in range(len(breaks_dict[team])-1):
            period_start = breaks_dict[team][i]
            period_end = breaks_dict[team][i+1]

            df_player_pass, df_player_location = get_passes_data(df_events_team, period_start['minute'], period_end['minute'])
            df_player_pass, df_player_location = filter_passes_data(df_player_pass, df_player_location, min_pass_count=selected_min_value)

            team_list.append({
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
        config_dict[team] = team_list

    for k in config_dict.keys():
        cols_all = len(config_dict[k])
        cols = st.columns(selected_num_cols_tab4)

        for i in range(cols_all):
            col = cols[i%selected_num_cols_tab4]

            period_start_time = config_dict[k][i]['period_start']['minute']
            period_end_time = config_dict[k][i]['period_end']['minute']
            period_start_type = config_dict[k][i]['period_start']['type']
            period_end_type = config_dict[k][i]['period_end']['type']
            sub_title = f"Passes from minute {period_start_time+1}-{period_end_time} ({period_start_type} - {period_end_type})"

            with col:
                plot_pass_map(
                    config_dict[k][i]['data_pass'],
                    config_dict[k][i]['data_location'],
                    title=config_dict[k][i]['team'],
                    sub_title=sub_title,
                    metric=config_dict[k][i]['metric'],
                    color=config_dict[k][i]['color'],
                    cmap_name=config_dict[k][i]['cmap']
                )

with tab5:
    pass


with tab6:
    pass


with tab7:
    pass


with tab8:
    st.markdown('### Team')
    st.markdown('### References')
    st.markdown("""
        - [Match Video](https://www.tokyvideo.com/video/netherlands-3-2-ukraine-full-match-euro-2020-group-stage-13-6-2021)
        - [Match Report: FBREF](https://fbref.com/en/matches/0e9919a5/Netherlands-Ukraine-June-13-2021-European-Championship)
        """)