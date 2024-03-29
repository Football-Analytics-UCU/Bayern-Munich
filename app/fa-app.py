import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd

from helper_passes import (get_passes_data, filter_passes_data, get_passes_player_data, create_pass_heatmap, plot_hulls, create_substitution_data, plot_pass_map, plot_pass_heatmap)
from helper_general import create_general_tab
from helper_goals import (get_shots_data, get_freeze, plot_all_shots, plot_shot_analysis)
import helper_corners

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "General", "Passes", "Player Passing Network", "Team Passing Network", "Goals", "Corners", "About"])


with tab1:
    create_general_tab(df_events)

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
    df_shots = get_shots_data(df_events)
    
    _, col501, _ = st.columns((0.1, 1, 0.1))
    with col501:
        st.markdown("## Total Shots")
        plot_all_shots(df_shots)
        
    _,col502,_ = st.columns((0.1, 1, 0.1))
    with col502:
        df_freeze = get_freeze(df_events)
        CHOICES = {
            "3fa18312-38f2-41d9-b94d-5133d515dc14": "51\'",
            "4d7ddff8-b418-43aa-a819-4e4a3b77ba61": "57\'",
            "91ccf9e2-8bd5-46e2-81cc-339cc5363854": "74\'",
            "f4973ccd-bee9-4e64-9750-1e2191f436d4": "78\'",
            "58013bf7-0a88-4a08-a1ad-ad90dd1ea68d": "84\'"
        }
        st.markdown('## Goals')
        input_select = st.selectbox('Select goal: ', CHOICES.keys(), index=2, format_func=lambda x: CHOICES[x], key=51)
        
        plot_shot_analysis(input_select, df_freeze, df_shots, df_lineups)


with tab6:
    df_lineups_tab_6 = df_lineups[['player_id', 'jersey_number', 'country']].copy()

    _, col701, _ = st.columns((0.25, 1, 0.25))
    with col701:
        #1
        IDS = ['bf6c9261-840b-4d8f-b7bd-9daa23b5b457']
        FF = ['corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'First corner')
        st.text('The player who get the ball out of bounds: Stefan de Vrij. Method: Clearance')
        #1 result
        IDS = ['80110e08-410a-4928-9601-3d2f0694589f'] 
        FF = ['freeze']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Result: Off target shot')
        #2
        IDS = ['49f48dc3-8bf0-41c8-af49-ab4ec4b164ee']
        FF = ['corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Second corner')
        st.text('The player who get the ball out of bounds: Heorhii Bushchan. Method: Goalkeeper saves')
        #3
        IDS = ['022d0dc4-9779-4224-9533-ef690e560907']
        FF = ['corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Third corner')
        st.text('The player who get the ball out of bounds: Mykola Matviyenko. Method: Clearance')
        #3 result
        IDS = ['9f9d09f9-c091-4c58-b7f0-873df2649cbd', '022d0dc4-9779-4224-9533-ef690e560907']
        FF = ['freeze']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Result: Off target shot')
        #4
        IDS = ['cfafc900-4c4d-49d5-b11c-8a3e0390eeca', '7545d8df-3938-4662-b1e0-cf45bb49b338']
        FF = ['solo', 'corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Fourth corner')
        st.text('The player who get the ball out of bounds: Illia Zabarnyi. Method: Clearance')
        #5
        IDS = ['9b62f73f-60d0-4cf2-bef1-7aeee121f6af', 'ee9667f9-4f49-4c61-890a-91685aef779b']
        FF = ['solo', 'corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Fifth corner')
        st.text('The player who get the ball out of bounds: Oleksandr Karavaev. Method: Interception')
        #6
        IDS = ['39755e7b-b68e-46f4-9fd9-07ac6b41ef2c']
        FF = ['corner']
        helper_corners.make_graph(df_events, df_lineups_tab_6, IDS, FF, 'Sixth corner')
        st.text('The player who get the ball out of bounds: Vitalii Mykolenko. Method: Duel')
    
    # Mean corner
    _, col601, _ = st.columns((0.25, 1, 0.25))    
    with col601:
        st.markdown('## Tournament Corners')
        values = [12.0, 6.5, 8.666666666666666, 9.166666666666666, 9.666666666666666, 5.833333333333333, 8.375, 14.25, 9.0, 8.0]
        v_names = ['A', 'B', 'C', 'D', 'E', 'F', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final']
        helper_corners.make_chart_bart(values, v_names, 2)


with tab7:
    st.markdown('#### Team Bayern')
    st.markdown("""
        - Ivaniuk Petro - Passing Network Tabs, @PetroIvaniuk
        - Sarana Maksym - Tab General, @Polosot
        - Bondarenko Olena - Tab Goals, @olena-bond
        - Yelisieiev Yura  - Tab Passes, @YuraYelisieiev
        - Petrov Roman - Tab Corners, @pingmar
        """)
    st.markdown('#### References')
    st.markdown("""
        - [Data: Statsbomb](https://github.com/statsbomb/open-data)
        - [Match Video](https://www.tokyvideo.com/video/netherlands-3-2-ukraine-full-match-euro-2020-group-stage-13-6-2021)
        - [Match Report: FBREF](https://fbref.com/en/matches/0e9919a5/Netherlands-Ukraine-June-13-2021-European-Championship)
        - [Mplsoccer](https://mplsoccer.readthedocs.io/en/latest/gallery/)
        - [Goal Plot: Mplsoccer](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_shot_freeze_frame.html#sphx-glr-gallery-pitch-plots-plot-shot-freeze-frame-py)
        - [Interactive Passing Networks](https://karun.in/blog/interactive-passing-networks.html)
        - [Creating Passmaps in Python](https://sharmaabhishekk.github.io/projects/passmap)
        """)
 