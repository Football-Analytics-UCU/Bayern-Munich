import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
from mplsoccer.pitch import VerticalPitch
from scipy.spatial import ConvexHull, Voronoi, Delaunay


def get_passes_data(df_events, team, minute_start, minute_end):
    """
    """
    df_events_temp = df_events[
        (df_events['type'] == 'Pass') &
        (df_events['pass_outcome'].isna()) &
        (df_events['team'] == team) &
        (df_events['minute'] > minute_start) &
        (df_events['minute'] <= minute_end)
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

    df_player_pass = df_events_temp.groupby(['player', 'pass_recipient']) \
        .agg(passes=('pass_start_x', 'size')).reset_index()

    df_player_pass = df_player_pass.merge(
        df_player_location[['player', 'x', 'y']], left_on='player', right_on='player'
    ).rename(columns={'x': 'passer_x', 'y': 'passer_y'}) \
        .merge(
        df_player_location[['player', 'x', 'y']], left_on='pass_recipient', right_on='player'
    ).rename(columns={'x': 'recipient_x', 'y': 'recipient_y', 'player_x': 'player'}) \
        .sort_values("passes", ascending=True)
    df_player_pass.drop('player_y', axis=1, inplace=True)

    return df_player_pass, df_player_location


def filter_passes_data(df_player_pass, df_player_location, min_pass_count=2):
    """
    """
    if min_pass_count != 1:
        df_player_pass = df_player_pass[df_player_pass['passes'] >= min_pass_count].copy().reset_index(drop=True)
        pass_player_list = list(set(df_player_pass['player'].tolist() + df_player_pass['pass_recipient'].tolist()))
        df_player_location = df_player_location[df_player_location['player'].isin(pass_player_list)].reset_index(
            drop=True)

    if not df_player_pass.empty:
        df_player_pass['pair'] = df_player_pass.apply(lambda x: x['player'] + '-' + x['pass_recipient'], axis=1)
        pair_unique = list(set(df_player_pass['pair'].to_list()))
        df_player_pass['is_pair'] = df_player_pass.apply(
            lambda x: x['pass_recipient'] + '-' + x['player'] in pair_unique, axis=1)

        df_player_pass['passes_scaled'] = df_player_pass['passes'].apply(lambda x: x / df_player_pass['passes'].max())
        df_player_location['passes_scaled'] = df_player_location['passes'].apply(
            lambda x: x / df_player_location['passes'].max())

        return df_player_pass, df_player_location
    else:
        return df_player_pass, df_player_location


class AnnotationHandler(HandlerLine2D):
    """
    Copied this from https://stackoverflow.com/a/49262926 
    Useful to add a annotation entry to legend since it is not automatically added
    """

    def __init__(self, ms, *args, **kwargs):
        self.ms = ms
        HandlerLine2D.__init__(self, *args, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        xdata, _ = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(np.array(xdata).shape, float)
        legline = FancyArrowPatch(posA=(xdata[0], ydata[0]),
                                  posB=(xdata[-1], ydata[-1]),
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
                             xy=(x1 + ((x2 - x1) / 2), y1 + ((y2 - y1) / 2)),
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
                x1=row.passer_y + y_shift,
                y1=row.passer_x + x_shift,
                x2=row.recipient_y + y_shift,
                y2=row.recipient_x + x_shift,
                ax=ax,
                color=cmap(row.passes_scaled),
                alpha=row.passes_scaled,
                lw=row.passes_scaled * 2,
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
                s=row.passes_scaled * 750,
                fc='white',
                ec=color,
                lw=3,
                zorder=10,
                label=f"Size indicates total passes made by player. \
                        \nMax Passes: {metric['max_passes']}. Min Passes: {metric['min_passes']}" if LABEL else ""
            )
            text = ax.text(
                row.y,
                row.x - 5,
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
                f"Passes from minute {period_start['minute'] + 1}-{period_end['minute']} ({period_start['type']} - {period_end['type']})",
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

        ax.legend(handles=h + [annotate],
                  handler_map={type(annotate): AnnotationHandler(5)},
                  loc=3, bbox_to_anchor=(0, -0.05))

        st.pyplot(fig, use_container_width=True)


def plot_pass_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax)
    st.pyplot(fig, use_container_width=True)


def draw_pitch(x_min=0,
               x_max=106,
               y_min=0,
               y_max=68,
               pitch_color="w",
               line_color="grey",
               line_thickness=1.5,
               point_size=20,
               orientation="horizontal",
               aspect="full",
               ax=None
               ):
    if orientation.lower().startswith("h"):
        first = 0
        second = 1
        arc_angle = 0

        if aspect == "half":
            ax.set_xlim(x_max / 2, x_max + 5)

    elif orientation.lower().startswith("v"):
        first = 1
        second = 0
        arc_angle = 90

        if aspect == "half":
            ax.set_ylim(x_max / 2, x_max + 5)


    else:
        raise NameError("You must choose one of horizontal or vertical")

    ax.axis("off")

    rect = plt.Rectangle((x_min, y_min),
                         x_max, y_max,
                         facecolor=pitch_color,
                         edgecolor="none",
                         zorder=-2)

    ax.add_artist(rect)

    x_conversion = x_max / 100
    y_conversion = y_max / 100

    pitch_x = [0, 5.8, 11.5, 17, 50, 83, 88.5, 94.2, 100]  # pitch x markings
    pitch_x = [x * x_conversion for x in pitch_x]

    pitch_y = [0, 21.1, 36.6, 50, 63.2, 78.9, 100]  # pitch y markings
    pitch_y = [x * y_conversion for x in pitch_y]

    goal_y = [45.2, 54.8]  # goal posts
    goal_y = [x * y_conversion for x in goal_y]

    # side and goal lines
    lx1 = [x_min, x_max, x_max, x_min, x_min]
    ly1 = [y_min, y_min, y_max, y_max, y_min]

    # outer boxed
    lx2 = [x_max, pitch_x[5], pitch_x[5], x_max]
    ly2 = [pitch_y[1], pitch_y[1], pitch_y[5], pitch_y[5]]

    lx3 = [0, pitch_x[3], pitch_x[3], 0]
    ly3 = [pitch_y[1], pitch_y[1], pitch_y[5], pitch_y[5]]

    # goals
    lx4 = [x_max, x_max + 2, x_max + 2, x_max]
    ly4 = [goal_y[0], goal_y[0], goal_y[1], goal_y[1]]

    lx5 = [0, -2, -2, 0]
    ly5 = [goal_y[0], goal_y[0], goal_y[1], goal_y[1]]

    # 6 yard boxes
    lx6 = [x_max, pitch_x[7], pitch_x[7], x_max]
    ly6 = [pitch_y[2], pitch_y[2], pitch_y[4], pitch_y[4]]

    lx7 = [0, pitch_x[1], pitch_x[1], 0]
    ly7 = [pitch_y[2], pitch_y[2], pitch_y[4], pitch_y[4]]

    # Halfway line, penalty spots, and kickoff spot
    lx8 = [pitch_x[4], pitch_x[4]]
    ly8 = [0, y_max]

    lines = [
        [lx1, ly1],
        [lx2, ly2],
        [lx3, ly3],
        [lx4, ly4],
        [lx5, ly5],
        [lx6, ly6],
        [lx7, ly7],
        [lx8, ly8],
    ]

    points = [
        [pitch_x[6], pitch_y[3]],
        [pitch_x[2], pitch_y[3]],
        [pitch_x[4], pitch_y[3]]
    ]

    circle_points = [pitch_x[4], pitch_y[3]]
    arc_points1 = [pitch_x[6], pitch_y[3]]
    arc_points2 = [pitch_x[2], pitch_y[3]]

    for line in lines:
        ax.plot(line[first], line[second],
                color=line_color,
                lw=line_thickness,
                zorder=-1)

    for point in points:
        ax.scatter(point[first], point[second],
                   color=line_color,
                   s=point_size,
                   zorder=-1)

    circle = plt.Circle((circle_points[first], circle_points[second]),
                        x_max * 0.088,
                        lw=line_thickness,
                        color=line_color,
                        fill=False,
                        zorder=-1)

    ax.add_artist(circle)

    arc1 = Arc((arc_points1[first], arc_points1[second]),
               height=x_max * 0.088 * 2,
               width=x_max * 0.088 * 2,
               angle=arc_angle,
               theta1=128.75,
               theta2=231.25,
               color=line_color,
               lw=line_thickness,
               zorder=-1)

    ax.add_artist(arc1)

    arc2 = Arc((arc_points2[first], arc_points2[second]),
               height=x_max * 0.088 * 2,
               width=x_max * 0.088 * 2,
               angle=arc_angle,
               theta1=308.75,
               theta2=51.25,
               color=line_color,
               lw=line_thickness,
               zorder=-1)

    ax.add_artist(arc2)
    ax.set_aspect("equal")
    return ax


def mode_including_nan(series):
    series_filled = series.fillna('NaN')
    mode_value = series_filled.mode()[0]
    if mode_value == 'NaN':
        return np.nan
    return mode_value


def create_pass_heatmap(data, team, n_clusters=40):
    pass_data = data[(data['type'] == "Pass") & (data['team'] == team)].reset_index(drop=True)

    pass_data['location_x'] = pass_data['location'].apply(lambda x: x[0])
    pass_data['location_y'] = pass_data['location'].apply(lambda x: x[1])
    pass_data['pass_end_location_x'] = pass_data['pass_end_location'].apply(lambda x: x[0])
    pass_data['pass_end_location_y'] = pass_data['pass_end_location'].apply(lambda x: x[1])

    background = '#F7F7F7'
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams.update({'font.size': 18})

    fig, ax = plt.subplots(figsize=(16.5, 10.5))
    fig.set_facecolor(background)
    ax.patch.set_facecolor(background)

    pitch = draw_pitch(x_min=0,
                       x_max=120,
                       y_min=0,
                       y_max=80,
                       orientation='horizontal',
                       aspect='fll',
                       pitch_color='#F7F7F7',
                       line_color='#3B3B3B',
                       ax=ax
                       )

    kde = sns.kdeplot(pass_data,
                      x='location_x',
                      y='location_y',
                      shade=True,
                      shade_lowest=False,
                      alpha=.5,
                      n_levels=20,
                      cmap='Reds'
                      )

    t_kmean = KMeans(n_clusters).fit(
        pass_data[['location_x', 'location_y', 'pass_end_location_x', 'pass_end_location_y']])
    pass_data['cluster'] = t_kmean.labels_

    pass_clusters = pass_data.groupby('cluster').agg(
        location_x=('location_x', 'mean'),
        location_y=('location_y', 'mean'),
        pass_end_location_x=('pass_end_location_x', 'mean'),
        pass_end_location_y=('pass_end_location_y', 'mean'),
        pass_outcome=('pass_outcome', mode_including_nan),
        count=('cluster', 'size')
    ).reset_index()

    for i in range(len(pass_clusters['location_x'])):
        try:
            if pd.isnull(pass_clusters['pass_outcome'][i]):
                plt.plot((pass_clusters['location_x'][i], pass_clusters['pass_end_location_x'][i]),
                         (pass_clusters['location_y'][i], pass_clusters['pass_end_location_y'][i]), color='green')
                plt.scatter(pass_clusters['location_x'][i], pass_clusters['location_y'][i], color='green')
            if pd.notnull(pass_clusters['pass_outcome'][i]):
                plt.plot((pass_clusters['location_x'][i], pass_clusters['pass_end_location_x'][i]),
                         (pass_clusters['location_y'][i], pass_clusters['pass_end_location_y'][i]), color='red')
                plt.scatter(pass_clusters['location_x'][i], pass_clusters['location_y'][i], color='red')
        except Exception as e:
            pass

    plt.xlim(0 - 5, 125)
    plt.ylim(0 - 5, 85)

    plt.figtext(
        0.055,
        0.96,
        f'{team} pass heatmap',
        fontsize=25,
        fontweight='bold',
        color='black'
    )
    plt.tight_layout()
    st.pyplot(fig)


def plot_hulls(team_dataset, lineups, team, main_color):
    unique_players = team_dataset['player'].unique()
    n_rows = (len(unique_players) + 3) // 4

    fig, axs = plt.subplots(n_rows, 4, figsize=(20, 18))
    axs = axs.flatten()

    for idx, player in enumerate(unique_players):
        if type(player) != str:
            continue
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=None, corner_arcs=True)
        pitch.draw(axs[idx])

        xmin, xmax, ymin, ymax = pitch.extent
        axs[idx].xaxis.set_ticks([xmin, xmax])
        axs[idx].yaxis.set_ticks([ymin, ymax])
        axs[idx].tick_params(labelsize=15)
        axs[idx].set_title(player, fontsize=18, pad=15)

        pos = list(lineups[lineups['player_name'] == player]['positions'].to_dict().values())
        if len(pos):
            pos = pos[0][0]['position']
        else:
            pos = ''
        axs[idx].text(85, 58.0, pos, size=12, verticalalignment='center', rotation=270)

        player_data = team_dataset[(team_dataset['player'] == player) & (team_dataset['type'] == 'Pass')]
        player_data['x'] = player_data['location'].apply(lambda x: x[0])
        player_data['y'] = player_data['location'].apply(lambda x: x[1])

        q_res = player_data[['x', 'y']].quantile([0.25, 0.75])
        IQR = q_res.loc[0.75] - q_res.loc[0.25]
        lb = q_res.loc[0.25] - 1.5 * IQR
        ub = q_res.loc[0.75] + 1.5 * IQR

        filt_pos = player_data[(player_data['x'] >= lb['x']) & (player_data['x'] <= ub['x']) &
                               (player_data['y'] >= lb['y']) & (player_data['y'] <= ub['y'])]

        pos_xy = filt_pos[['x', 'y']].values
        if len(filt_pos) > 2:
            ch = ConvexHull(filt_pos[['x', 'y']])
            axs[idx].scatter(filt_pos.y, filt_pos.x, color=main_color, s=100)

            for i in ch.simplices:
                axs[idx].plot(pos_xy[i, 1], pos_xy[i, 0], 'black')
                axs[idx].fill(pos_xy[ch.vertices, 1], pos_xy[ch.vertices, 0], alpha=0.1)

    fig.suptitle(f"{team} players Convex Hulls", fontsize=25, verticalalignment='top', y=1.01)
    plt.tight_layout()
    axs[0].set_axis_off()
    axs[-2].set_axis_off()
    axs[-1].set_axis_off()
    st.pyplot(fig)
