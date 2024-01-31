import streamlit as st
import pandas as pd
from mplsoccer.pitch import Pitch
import ast
import matplotlib.pyplot as plt


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
        ax = pd.DataFrame({'a': [1, 0.004, 1]}).T.plot.barh(stacked=True, figsize=(10, 1),
                                                            color=['lightgray', 'white', 'lightgray'])
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
                labels=[
                    {'%': f'{cc:.1%} - {lb}', 'N': f'{cc:.0f} - {lb}', 'F': f'{cc:.2f} - {lb}', 'Z': f'0 - {lb}'}[
                        format] for cc in c.datavalues],
                color='white'
            )

    fig = ax.get_figure()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    return fig


def draw_field(df_events, command1, command2):
    def plot_players(pitch, lineup, color, command, opposite=False, omit_players=0):
        COORDS = {
            'Goalkeeper': (5, 50),
            'Center Attacking Midfield': (37, 50),
            'Center Back': (17, 50),
            'Center Defensive Midfield': (32, 50),
            'Center Forward': (46, 50),
            'Left Back': (17, 83),
            'Left Center Back': (17, 67),
            'Left Center Forward': (45, 77),
            'Left Center Midfield': (32, 83),
            'Left Defensive Midfield': (32, 67),
            'Left Wing': (41, 17),
            'Left Wing Back': (23, 85),
            'Right Back': (17, 17),
            'Right Center Back': (17, 33),
            'Right Center Forward': (45, 23),
            'Right Center Midfield': (32, 17),
            'Right Defensive Midfield': (32, 33),
            'Right Wing': (41, 83),
            'Right Wing Back': (23, 15)
        }

        lineup_coords = [COORDS[x['position']['name']] for x in lineup]
        lineup_labels = [x['player']['name'] for x in lineup]

        for xy, t, lb in zip(lineup_coords, [str(x['jersey_number']) for x in lineup], lineup_labels):
            pitch.scatter(100 - xy[0] if opposite else xy[0], xy[1], s=300, c=color, label=f"{t} {lb}", ax=ax)
            pitch.annotate(
                xy=(100 - xy[0] - 0.15 - len(t) * 0.3 if opposite else xy[0] - 0.15 - len(t) * 0.3, xy[1] - 0.7),
                text=t, ax=ax)

        h, l = ax.get_legend_handles_labels()
        plt.rcParams['legend.title_fontsize'] = 20
        legend = plt.legend(h[omit_players:],
                            l[omit_players:],
                            title=r"$\bf{" + command + "}$", edgecolor='#22312b',
                            facecolor='#22312b', labelcolor='#edede9', fontsize=15,
                            loc='upper right' if opposite else 'upper left',
                            bbox_to_anchor=(1.38, 1) if opposite else (-0.38, 1),
                            borderaxespad=0.8
                            )
        legend.get_title().set_color('#edede9')
        ax.add_artist(legend)

    lineup = df_events.iloc[0:2, :].loc[:, ['team', 'tactics']].set_index('team').to_dict()['tactics']

    pitch = Pitch(pitch_type='opta', pitch_color='forestgreen', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(16, 10), constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#22312b')

    plt.subplots_adjust(right=1.4, left=1.4, top=1.4)

    plot_players(pitch, lineup[command1]['lineup'], 'orange', opposite=False, command=command1)
    plot_players(pitch, lineup[command2]['lineup'], 'dodgerblue', opposite=True, command=command2,
                 omit_players=len(lineup[command1]['lineup']))

    ax.set_title(
        f'{command1} ({"-".join(str(lineup[command1]["formation"]))}) vs {command2} ({"-".join(str(lineup[command2]["formation"]))})',
        fontsize=25, color='#edede9')

    return fig


def get_column_values(df_events, column, mapping, reverse_index=False):

    grouped_df = df_events.pivot_table(
        values=['id'],
        index=['team'],
        columns=[column],
        aggfunc='count', fill_value=0
    )
    grouped_df.columns = grouped_df.columns.droplevel()

    for c in mapping:
        if c not in grouped_df.columns:
            grouped_df[c] = 0

    if '_total_' in mapping:
        grouped_df['_total_'] = grouped_df.sum(axis=1)

    if reverse_index:
        grouped_df.index = [{'Ukraine': 'Netherlands', 'Netherlands': 'Ukraine'}[i] for i in grouped_df.index]

    return grouped_df[list(mapping)].rename(columns=mapping)


def get_data(df_events):
    RENAME = {'possession': 'Possession'}

    res_df = df_events.groupby('team').agg({'possession': "sum"})

    foul_committed_card = get_column_values(
        df_events, 'foul_committed_card', mapping={'Red Card': 'Red cards', 'Yellow Card': 'Yellow cards'}
    )

    event_types = get_column_values(
        df_events, 'type', mapping={'Pass': 'Passes attempted', 'Foul Committed': 'Fouls committed'}
    )

    goalkeeper_types = get_column_values(
        df_events, 'goalkeeper_type', mapping={'Goal Conceded': 'Goals', '_total_': 'Total attempts',
                                               'Collected': 'Collected', 'Keeper Sweeper': 'Keeper Sweeper',
                                               'Shot Faced': 'Shot Faced', 'Shot Saved': 'Shot Saved',
                                               'Smother': 'Smother'},
        reverse_index=True
    )

    duel_types = get_column_values(
        df_events, 'duel_type', mapping={'Aerial Lost': 'Aerial losts', 'Tackle': 'Tackles'}
    )

    res_df = pd.concat([res_df, foul_committed_card, event_types, goalkeeper_types, duel_types], axis=1).fillna(0)

    return res_df.rename(columns=RENAME)


def create_general_tab(df_events):

    data = get_data(df_events)

    st.title("Lineup")

    st.pyplot(draw_field(df_events, "Netherlands", "Ukraine"))

    st.title("Performance")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Possession", format='%'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Passes attempted", format='N'))

    st.title("Attacking")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Goals", format='N'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Total attempts", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Collected", format='N'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Keeper Sweeper", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Shot Faced", format='N'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Shot Saved", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Smother", format='N'))

    st.title("Duels")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Aerial losts", format='N'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Tackles", format='N'))

    st.title("Disciplinary")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Yellow cards", format='N'))
        with col2:
            st.pyplot(get_binary_chart_wrapper(data, "Red cards", format='N'))

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(get_binary_chart_wrapper(data, "Fouls committed", format='N'))
