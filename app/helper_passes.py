import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mplsoccer.pitch import VerticalPitch
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch


def get_passes_data(df_events, team, minute_start, minute_end):
    """
    """
    df_events_temp = df_events[
        (df_events['type']=='Pass')&
        (df_events['pass_outcome'].isna())&
        (df_events['team']==team)&
        (df_events['minute']>minute_start)&
        (df_events['minute']<=minute_end)
    ].copy().reset_index(drop=True)

    df_events_temp[['pass_start_x', 'pass_start_y']] = pd.DataFrame(
        df_events_temp["location"].values.tolist(), index=df_events_temp.index)
    df_events_temp[['pass_end_x', 'pass_end_y']] = pd.DataFrame(
        df_events_temp["pass_end_location"].tolist(), index=df_events_temp.index)

    df_player_location = df_events_temp.groupby(['player']).agg(
        x=('pass_start_x', 'mean'), 
        y=('pass_start_y', 'mean'),
        passes=('pass_start_x', 'size')
    ).reset_index().sort_values("passes", ascending=True)

    df_player_pass = df_events_temp.groupby(['player', 'pass_recipient'])\
                                   .agg(passes=('pass_start_x', 'size')).reset_index()

    df_player_pass = df_player_pass.merge(
        df_player_location[['player', 'x', 'y']], left_on='player', right_on='player'
    ).rename(columns={'x': 'passer_x', 'y': 'passer_y'})\
                                   .merge(
        df_player_location[['player', 'x', 'y']], left_on='pass_recipient', right_on='player'
    ).rename(columns={'x': 'recipient_x', 'y': 'recipient_y', 'player_x': 'player'})\
                                   .sort_values("passes", ascending=True)
    df_player_pass.drop('player_y', axis=1, inplace=True)

    return df_player_pass, df_player_location

def filter_passes_data(df_player_pass, df_player_location, min_pass_count=2):
    """
    """
    if min_pass_count!=1:
        df_player_pass = df_player_pass[df_player_pass['passes']>=min_pass_count].copy().reset_index(drop=True)
        pass_player_list = list(set(df_player_pass['player'].tolist() + df_player_pass['pass_recipient'].tolist()))
        df_player_location = df_player_location[df_player_location['player'].isin(pass_player_list)].reset_index(drop=True)

    if not df_player_pass.empty:
        df_player_pass['pair'] = df_player_pass.apply(lambda x: x['player']+'-'+x['pass_recipient'], axis=1)
        pair_unique = list(set(df_player_pass['pair'].to_list()))
        df_player_pass['is_pair'] = df_player_pass.apply(lambda x: x['pass_recipient'] + '-' + x['player'] in pair_unique, axis=1)

        df_player_pass['passes_scaled'] = df_player_pass['passes'].apply(lambda x: x/df_player_pass['passes'].max())
        df_player_location['passes_scaled'] = df_player_location['passes'].apply(lambda x: x/df_player_location['passes'].max())

        return df_player_pass, df_player_location
    else:
        return df_player_pass, df_player_location

class AnnotationHandler(HandlerLine2D):
    """
    Copied this from https://stackoverflow.com/a/49262926 
    Useful to add a annotation entry to legend since it is not automatically added
    """
    def __init__(self,ms,*args,**kwargs):
        self.ms = ms
        HandlerLine2D.__init__(self,*args,**kwargs)
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        xdata, _ = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(np.array(xdata).shape, float)
        legline = FancyArrowPatch(posA=(xdata[0],ydata[0]),
                                  posB=(xdata[-1],ydata[-1]),
                                  mutation_scale=self.ms,
                                  **orig_handle.arrowprops)
        legline.set_transform(trans)
        return legline,

def add_arrow(x1, y1, x2, y2, ax, text_label, **kwargs):
    """
    Helper function to add an arrow b/w two points
    defined by (x1, y1) and (x2, y2)

    A line is drawn from point A to point B. An arrow is then drawn from point A along halfway to point B.
    This is an easy way to get an arrow with a head in the middle
    """
    ax.plot([x1, x2], [y1, y2], **kwargs)
        
    annotation = ax.annotate("", 
        xytext=(x1, y1), 
        xy=(x1+((x2-x1)/2), y1+((y2-y1)/2)),
        arrowprops=dict(arrowstyle="->", **kwargs),
        zorder=10,
        size=30,
        label=text_label
    )
    return annotation

def plot_pass_map(df_player_pass, df_location, team_name, metric, period_start, period_end, color, cmap_name):
    """
    Create a passmap for a single team 
    """
    ARROW_SHIFT = 2
    cmap = plt.cm.get_cmap(cmap_name)

    fig, ax = VerticalPitch().draw(nrows=1, ncols=1, figsize=(16, 10))
    fig.set_facecolor("white")
    
    if not df_player_pass.empty:
        # draw arrows
        annotations = []
        for row in df_player_pass.itertuples():
            if row.is_pair:
                if abs(row.recipient_y - row.passer_y) > abs(row.recipient_x - row.passer_x):
                    if row.player > row.pass_recipient:
                        x_shift, y_shift = 0, ARROW_SHIFT
                    else: 
                        x_shift, y_shift = 0, -ARROW_SHIFT
                else:
                    if row.player > row.pass_recipient:
                        x_shift, y_shift = ARROW_SHIFT, 0
                    else: 
                        x_shift, y_shift = -ARROW_SHIFT, 0
            else:
                x_shift = 0
                y_shift = 0

            arrow = add_arrow(
                x1=row.passer_y+y_shift,
                y1=row.passer_x+x_shift,
                x2=row.recipient_y+y_shift, 
                y2=row.recipient_x+x_shift,
                ax=ax, 
                color=cmap(row.passes_scaled), 
                alpha=row.passes_scaled,
                lw=row.passes_scaled*2,
                text_label=f"Darker color indicates higher number of passes in that direction.\
                             \nMax Passes in direction: {metric['max_passes_direction']}.\
                             \nMin Passes in direction: {metric['min_passes_direction']}."
            )
            annotations.append(arrow)
        
        # draw locations
        LABEL = True
        texts = []
        for row in df_location.itertuples():
            ax.scatter(
                row.y, 
                row.x, 
                s=row.passes_scaled*750,
                fc='white',
                ec=color,
                lw=3,
                zorder=10, 
                label=f"Size indicates total passes made by player. \
                        \nMax Passes: {metric['max_passes']}. Min Passes: {metric['min_passes']}" if LABEL else ""
            )
            text = ax.text(
                row.y, 
                row.x-5, 
                s=row.player, 
                ha='center', 
                va='center', 
                zorder=20
            )
            text.set_path_effects([pe.PathPatchEffect(offset=(2, -2), hatch='xxxx', facecolor='gray')])
            texts.append(text)
            LABEL = False

        # drow annotations: title, subtitle
        title = ax.text(0, 130, 
            team_name, 
            color=color,
            fontsize=24, 
            va='top')
        ax.text(0, 124, 
            f"Passes from minute {period_start['minute']+1}-{period_end['minute']} ({period_start['type']} - {period_end['type']})",
            fontsize=12, 
            va='top')
        # ax.text(1, 10, f"Min Passes: {metric['min_passes']}", fontsize=12, va='top', ha='left')

        title.set_path_effects([
            pe.PathPatchEffect(offset=(1, -1), hatch='xxxx', facecolor='black'),
            pe.PathPatchEffect(edgecolor='black', linewidth=.8, facecolor=color)
        ])

        # add legend for annotations
        h, _ = ax.get_legend_handles_labels()
        annotate = annotations[-1]

        ax.legend(handles = h + [annotate], 
                  handler_map={type(annotate) : AnnotationHandler(5)}, 
                  loc=3, bbox_to_anchor=(0, -0.05))
        
        st.pyplot(fig, use_container_width=True)

def plot_pass_heatmap(df):
	fig, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(df, annot=True, linewidths=.5, ax=ax)
	st.pyplot(fig, use_container_width=True)