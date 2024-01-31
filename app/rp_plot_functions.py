import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch, FontManager, Pitch
import streamlit as st
import matplotlib.patches as mpatches

def make_graph(ids, ff, title, padding=[None,None,None,None]):
  pitch = VerticalPitch(goal_type='box', half=True, pad_left = padding[0], pad_right = padding[1], pad_top = padding[2], pad_bottom = padding[3])
  fig, axs = pitch.grid(figheight=8, endnote_height=0,
                        title_height=0.1, title_space=0.02,
                        axis=False,
                        grid_height=0.83)
  for ID, tp in zip(ids, ff): 
    if tp == 'freeze':
      df_shot_event = events[events.id == ID].dropna(axis=1, how='all').copy()
      df_freeze_frame = pd.json_normalize(df_shot_event.shot_freeze_frame.dropna().iloc[0], sep='_') #freeze[freeze.id == ID].copy()
      df_freeze_frame[['x','y']] = pd.DataFrame(df_freeze_frame.location.tolist(), index=df_freeze_frame.index)
      df_freeze_frame = df_freeze_frame.merge(df_lineup, how='left', on='player_id')
      team1 = df_shot_event.team.iloc[0]
      team2 = 'Netherlands' if team1 == 'Ukraine' else 'Ukraine'
      COLOR_1 = COLOR_U if team1 == 'Ukraine' else COLOR_N
      COLOR_2 = COLOR_U if team2 == 'Ukraine' else COLOR_N
      df_team1 = df_freeze_frame[df_freeze_frame.teammate == True]
      df_team2 = df_freeze_frame[df_freeze_frame.teammate == False]
      pitch.goal_angle(df_shot_event.location.item()[0], df_shot_event.location.item()[-1], ax=axs['pitch'], alpha=0.2, zorder=1.1,
                  color='#cb5a4c', goal='right')
      pitch.lines(df_shot_event.location.item()[0], df_shot_event.location.item()[-1],
                    df_shot_event.shot_end_location.item()[0], df_shot_event.shot_end_location.item()[1], comet=True,
                    label='shot', color='#cb5a4c', ax=axs['pitch'])
      pitch.scatter(df_team1.x, df_team1.y, s=600, c=COLOR_1, label='Attacker', ax=axs['pitch'])
      pitch.scatter(df_team2.x, df_team2.y, s=600, c=COLOR_2, label='Defender', ax=axs['pitch'])
      pitch.scatter(df_shot_event.location.item()[0], df_shot_event.location.item()[-1], c=COLOR_1, marker='football',
                      s=600, ax=axs['pitch'], label=f'Shooter: {df_shot_event.player.item()}', zorder=1.2)
      for i, (label, name) in enumerate(zip(df_freeze_frame.jersey_number, df_freeze_frame.player_name)):
        pitch.annotate(label, (df_freeze_frame.x[i], df_freeze_frame.y[i]),
                   va='center', ha='center', color='white', fontsize=15, ax=axs['pitch'])
        pitch.text(50+2*i, -20, f'{label}: {name}', va='center', ha='center', ax=axs['pitch'])
    elif tp == 'corner':
      df_shot_event = events[events.id == ID].dropna(axis=1, how='all').copy()
      team1 = df_shot_event.team.iloc[0]
      team2 = 'Netherlands' if team1 == 'Ukraine' else 'Ukraine'
      COLOR_1 = COLOR_U if team1 == 'Ukraine' else COLOR_N
      COLOR_2 = COLOR_U if team2 == 'Ukraine' else COLOR_N
      pitch.lines(df_shot_event.location.item()[0], df_shot_event.location.item()[-1],
                    df_shot_event.pass_end_location.item()[0], df_shot_event.pass_end_location.item()[-1], comet=True,
                    label='shot', color='#cb5a4c', ax=axs['pitch'])
      pitch.scatter(df_shot_event.location.item()[0], df_shot_event.location.item()[-1], c = COLOR_1, marker='football',
                      s=600, ax=axs['pitch'], label='Shooter', zorder=1.2)
      pitch.scatter(df_shot_event.pass_end_location.item()[0], df_shot_event.pass_end_location.item()[-1], s=600, c=COLOR_1, label='Reciver', ax=axs['pitch'])
      pitch.annotate(df_shot_event.player.item(), (df_shot_event.location.item()[0] - C, df_shot_event.location.item()[-1]),
                   va='top', ha='center', color='black', fontsize=10, ax=axs['pitch'])
      if 'pass_recipient' in df_shot_event.columns:
        pitch.annotate(df_shot_event.pass_recipient.item(), (df_shot_event.pass_end_location.item()[0] - C, df_shot_event.pass_end_location.item()[-1]),
                     va='top', ha='center', color='black', fontsize=10, ax=axs['pitch'])
      else:
        pitch.annotate('Unknown', (df_shot_event.pass_end_location.item()[0] - C, df_shot_event.pass_end_location.item()[-1]),
                     va='top', ha='center', color='black', fontsize=10, ax=axs['pitch'])
    else:
      df_shot_event = events[events.id == ID].dropna(axis=1, how='all').copy()
      team1 = df_shot_event.team.iloc[0]
      team2 = 'Netherlands' if team1 == 'Ukraine' else 'Ukraine'
      COLOR_1 = COLOR_U if team1 == 'Ukraine' else COLOR_N
      COLOR_2 = COLOR_U if team2 == 'Ukraine' else COLOR_N
      pitch.scatter(120-df_shot_event.location.item()[0], 80-df_shot_event.location.item()[-1], c = COLOR_1,
                      s=600, ax=axs['pitch'], label='Defender', zorder=1.2)
      pitch.annotate(df_shot_event.player.item(), (120-df_shot_event.location.item()[0] + C, 80-df_shot_event.location.item()[-1]),
                   va='center', ha='center', color='black', fontsize=10, ax=axs['pitch'])
  axs['title'].text(0.5, 0.5, f'{title}',
                    va='center', ha='center', color='black',
                    fontsize=25)
  #legend = axs['pitch'].legend(df_freeze_frame.player_name.to_list(), df_freeze_frame.jersey_number.to_list())
  legend = axs['pitch'].legend(loc='lower left', labelspacing=1.5)
  for text in legend.get_texts():
    text.set_fontsize(20)
    text.set_va('center')
  st.pyplot(fig)

def make_chart_bart(data, data_names, spec=0):
  fig, ax = plt.subplots()
  #fig.set_size_inches(18.5, 15.5)
  color = ['C0' for i in range(len(data))]
  red_patch = mpatches.Patch(color='red', label='Our Group')
  blue_patch = mpatches.Patch(color='C0', label='Others')
  y_pos = np.arange(len(data_names))
  if spec != 0:
    color[spec] = 'red'
  ax.barh(y_pos, data, align='center', height=0.8, color=color)
  ax.set_yticks(y_pos, labels=data_names)
  ax.invert_yaxis() 
  ax.set_xlabel('Num. of corner events')
  ax.set_title('Mean corner events for')
  plt.xticks(np.arange(0, max(data)+1, 1.0))
  plt.legend(loc="lower right", handles=[red_patch, blue_patch])
  st.pyplot(fig)

COLOR_U = 'blue'
COLOR_N = 'orange'
SIZE = 1
C = 2
competition_id=55
season_id=43
match_id=3788746
PATH_EVENT = f'app/data/events.pkl'
PATH_LINEUP = f'app/data/lineups.pkl'
events = pd.read_pickle(PATH_EVENT)
lineup = pd.read_pickle(PATH_LINEUP)
df_lineup = lineup[['player_id', 'jersey_number', 'country']].copy()
