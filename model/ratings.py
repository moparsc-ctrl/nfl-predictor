"""
Calcula off_strength / def_strength (por puntos) y su version ajustada con EPA,
igual que el modelo validado en el notebook de prueba.
"""
import pandas as pd


def build_points_ratings(games: pd.DataFrame, k_shrink: int = 6):
    """games: dataframe con home_team, away_team, home_score, away_score."""
    teams = pd.unique(games[["home_team", "away_team"]].values.ravel())
    league_avg_pts = pd.concat([games["home_score"], games["away_score"]]).mean()

    def team_stats(team):
        home = games[games["home_team"] == team]
        away = games[games["away_team"] == team]
        pts_for = pd.concat([home["home_score"], away["away_score"]])
        pts_against = pd.concat([home["away_score"], away["home_score"]])
        n = len(pts_for)
        pts_for_avg = pts_for.mean()
        pts_against_avg = pts_against.mean()
        pts_for_shrunk = (pts_for_avg * n + league_avg_pts * k_shrink) / (n + k_shrink)
        pts_against_shrunk = (pts_against_avg * n + league_avg_pts * k_shrink) / (n + k_shrink)
        return pts_for_shrunk, pts_against_shrunk, n

    stats = {t: team_stats(t) for t in teams}
    stats_df = pd.DataFrame(stats, index=["pts_for_avg", "pts_against_avg", "n_partidos"]).T
    stats_df["off_strength"] = stats_df["pts_for_avg"] / league_avg_pts
    stats_df["def_strength"] = stats_df["pts_against_avg"] / league_avg_pts
    return stats_df, league_avg_pts


def add_epa_ratings(stats_df: pd.DataFrame, pbp: pd.DataFrame, weight: float = 0.08):
    """Ajusta off_strength/def_strength con EPA por jugada. Devuelve stats_df con
    columnas off_strength_epa / def_strength_epa."""
    pbp_reg = pbp[(pbp["season_type"] == "REG") & (pbp["play_type"].isin(["pass", "run"]))]

    off_epa = pbp_reg.groupby("posteam")["epa"].mean().rename("off_epa_play")
    def_epa = pbp_reg.groupby("defteam")["epa"].mean().rename("def_epa_play")
    epa_df = pd.concat([off_epa, def_epa], axis=1)

    epa_off_z = (epa_df["off_epa_play"] - epa_df["off_epa_play"].mean()) / epa_df["off_epa_play"].std()
    epa_def_z = (epa_df["def_epa_play"] - epa_df["def_epa_play"].mean()) / epa_df["def_epa_play"].std()

    stats_df["off_strength_epa"] = stats_df["off_strength"] * (
        1 + epa_off_z.reindex(stats_df.index).fillna(0) * weight
    )
    stats_df["def_strength_epa"] = stats_df["def_strength"] * (
        1 + epa_def_z.reindex(stats_df.index).fillna(0) * weight
    )
    return stats_df
