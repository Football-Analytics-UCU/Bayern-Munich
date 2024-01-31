import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch


def get_shots_data(df_events):
    df_events[['x', 'y']] = df_events['location'].apply(pd.Series)
    df_events[['end_x', 'end_y', "h"]] = df_events['shot_end_location'].apply(pd.Series)
    df_shots = df_events.loc[df_events['type'] == "Shot",  
                         ['id','team','x', 'y', 'shot_outcome', 'minute', 'shot_statsbomb_xg', 'player', 'end_x', 'end_y']].set_index('id')
    return df_shots

def get_freeze(df_events):
    df_freeze = df_events.loc[(df_events['type'] == "Shot") & (df_events['shot_outcome'] == "Goal")]
    df_freeze = df_freeze[['shot_freeze_frame', 'id']].explode('shot_freeze_frame')
    df_res = pd.json_normalize(df_freeze['shot_freeze_frame'])
    df_res['id'] = df_freeze['id'].values
    df_res[['x', 'y']] = df_res['location'].apply(pd.Series)
    df_res = df_res.rename(columns={
    'player.id': 'player_id',
    'player.name': 'player_name',
    'position.id': 'position_id',
    'position.name': 'position_name'
})

    return df_res



def plot_all_shots (df_shots):
    """
    
    """

    df_netherlands = df_shots[df_shots['team'] == 'Netherlands']
    df_ukraine = df_shots[df_shots['team'] == 'Ukraine']


    pitch = Pitch()
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

    for i, row in df_netherlands.iterrows():
        if row["shot_outcome"] == 'Goal':
            pitch.scatter(row.x, row.y, alpha=1, s=500, marker='football', ax=ax['pitch'], label='Netherlands')
            pitch.annotate(str(row["minute"]) + "'", (row.x + 1, row.y - 2), ax=ax['pitch'], fontsize=12)
        else:
            pitch.scatter(row.x, row.y, alpha=0.4, s=500, c=["#eb9835"], ax=ax['pitch'])
        
    # Netherlands
    total_shots_nl = len(df_netherlands)
    on_target_nl = ((df_netherlands['shot_outcome'] == 'Saved') | (df_netherlands['shot_outcome'] == 'Goal')).sum()
    goals_nl = (df_netherlands['shot_outcome'] == 'Goal').sum()

    panel_width_nl = 0.2
    panel_height_nl = 0.3
    panel_rect_nl = plt.Rectangle((0, 0), panel_width_nl, panel_height_nl)
    ax['pitch'].add_patch(panel_rect_nl)

    panel_text_nl = f'Netherlands\nTotal Shots: {total_shots_nl}\nOn Target: {on_target_nl}\nGoals: {goals_nl}'
    ax['pitch'].text(0.79, 0.1, panel_text_nl, fontsize=18, color='black', transform=ax['pitch'].transAxes)

    # Ukraine
    total_shots_ukraine = len(df_ukraine)
    on_target_ukraine = ((df_ukraine['shot_outcome'] == 'Saved') | (df_ukraine['shot_outcome'] == 'Goal')).sum()
    goals_ukraine = (df_ukraine['shot_outcome'] == 'Goal').sum()

    panel_width_ukraine = 0.2
    panel_height_ukraine = 0.3
    panel_rect_ukraine = plt.Rectangle((0, 0), panel_width_ukraine, panel_height_ukraine)
    ax['pitch'].add_patch(panel_rect_ukraine)

    panel_text_ukraine = f'Ukraine\nTotal Shots: {total_shots_ukraine}\nOn Target: {on_target_ukraine}\nGoals: {goals_ukraine}'
    ax['pitch'].text(0.05, 0.1, panel_text_ukraine, fontsize=18, color='black', transform=ax['pitch'].transAxes)

    for i, row in df_ukraine.iterrows():
        if row["shot_outcome"] == 'Goal':
            pitch.scatter(120 - row.x, 80 - row.y, alpha=1, s=500, marker='football', ax=ax['pitch'], label='Ukraine')
            pitch.annotate(str(row["minute"]) + "'", (120 - row.x + 1, 80 - row.y - 2), ax=ax['pitch'], fontsize=12)
        else:
            pitch.scatter(120 - row.x, 80 - row.y, alpha=0.4, s=500, c=["#2442ef"], ax=ax['pitch'])

    fig.suptitle("All shots", fontsize=30)
    st.pyplot(fig, use_container_width=True)
    
    
    
def plot_shot_analysis(SHOT_ID, df_freeze, df_shots, df_lineups):
    df_freeze_frame = df_freeze[df_freeze.id == SHOT_ID].copy()
    df_shot_event = df_shots.loc[df_shots.index == SHOT_ID].dropna(axis=1, how='all').copy()

    # Add the jersey number
    df_freeze_frame = df_freeze_frame.merge(df_lineups, how='left', on='player_id')

    # Strings for team names
    team1 = "Netherlands"
    team2 = "Ukraine"

    # teams
    df_team1 = df_freeze_frame[df_freeze_frame.country == team1]
    df_team2 = df_freeze_frame[(df_freeze_frame.country == team2)]
                                    

    # Setup the pitch
    pitch = VerticalPitch(half=True, goal_type='box',pad_bottom=-20)

    # Plot the players
    fig, axs = pitch.grid(figheight=8, endnote_height=0, title_height=0.1, title_space=0.02,
                          axis=False, grid_height=0.83)

    sc1 = pitch.scatter(df_team1.x, df_team1.y, s=600, c='#eb9835', label=team1, ax=axs['pitch'])
    sc2 = pitch.scatter(df_team2.x, df_team2.y, s=600,
                        c='#2442ef', label=team2, ax=axs['pitch'])


    # Plot the shot
    sc3 = pitch.scatter(df_shot_event.x, df_shot_event.y, marker='football',
                        s=600, ax=axs['pitch'], label='Shooter', zorder=1.2)
    line = pitch.lines(df_shot_event.x, df_shot_event.y,
                       df_shot_event.end_x, df_shot_event.end_y, comet=True,
                       label='shot', color='#cb5a4c', ax=axs['pitch'])

    # Plot the angle to the goal
    pitch.goal_angle(df_shot_event.x, df_shot_event.y, ax=axs['pitch'],
                     alpha=0.2, zorder=1.1, color='#cb5a4c', goal='right')

    # Plot the jersey numbers
    for i, label in enumerate(df_freeze_frame.jersey_number):
        pitch.annotate(label, (df_freeze_frame.x[i], df_freeze_frame.y[i]),
                       va='center', ha='center', color='white',
                       fontsize=15, ax=axs['pitch'])

    # Add a panel on the right side displaying jersey numbers and player nicknames
    panel_ax = fig.add_axes([0.8, 0.1, 0.15, 0.8])
    panel_ax.axis('off')
    
    line_spacing = 0.05
    team1_rows = df_freeze_frame[df_freeze_frame['country'] == team1].sort_values('jersey_number')
    team2_rows = df_freeze_frame[df_freeze_frame['country'] == team2].sort_values('jersey_number')
    combined_rows = pd.concat([team1_rows, team2_rows])

    for i in range(len(combined_rows)):
        row = combined_rows.iloc[i]
        team_color = '#eb9835' if row['country'] == team1 else '#2442ef'
        panel_ax.text(0.5, i * line_spacing, f"{int(row['jersey_number'])} - {row['player_name_x']}",
                  va='center', ha='center', color=team_color, fontsize=9)


    # Add a legend and title
    legend = axs['pitch'].legend(loc='center left', labelspacing=1.5)
    for text in legend.get_texts():
        text.set_fontsize(20)
        text.set_va('center')

    axs['title'].text(0.5, 0.5, f'{df_shot_event.player.iloc[0]}\n{df_shot_event.minute.iloc[0]}\'',
                        va='center', ha='center', color='black',
                      fontsize=25)

    st.pyplot(fig, use_container_width=True)

    
    
    
    
    