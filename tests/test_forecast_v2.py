import math
from dataclasses import replace

import pandas as pd
import pytest

from fpl_app.logic.forecast_v2 import (
    DEFAULT_RATE_PRIOR_MINUTES,
    FORECAST_VERSION,
    Reliability,
    build_forecast_priors,
    expected_minutes_for_player,
    forecast_player_v2,
    forecast_players_v2,
    shrunk_rates_for_player,
    _expected_conceded_goal_blocks,
    _fixture_components,
)


def _player(
    player_id: int,
    *,
    team_id: int = 1,
    position: str = "MID",
    price: int = 75,
    minutes: float = 900.0,
    starts: float = 10.0,
    sample_matches: float = 10.0,
    xg_per90: float = 0.30,
    xa_per90: float = 0.20,
    bonus_per90: float = 0.20,
    cs_per90: float = 0.25,
    xgc_per90: float = 1.25,
    status: str = "a",
    chance: float | None = None,
) -> dict:
    exposure = max(0.0, minutes) / 90.0
    row = {
        "id": player_id,
        "team_id": team_id,
        "singular_name_short": position,
        "now_cost": price,
        "minutes": minutes,
        "starts": starts,
        "sample_matches": sample_matches,
        "expected_goals": xg_per90 * exposure,
        "expected_assists": xa_per90 * exposure,
        "bonus": bonus_per90 * exposure,
        "clean_sheets": cs_per90 * exposure,
        "expected_goals_conceded": xgc_per90 * exposure,
        "status": status,
    }
    if chance is not None:
        row["chance_of_playing_next_round"] = chance
    return row


def _fixtures(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=(
            "event",
            "home_team",
            "away_team",
            "home_fdr",
            "away_fdr",
        ),
    )


def _one_fixture(event: int = 5, *, fdr: int = 3, home: bool = True) -> pd.DataFrame:
    if home:
        return _fixtures([(event, 1, 2, fdr, 3)])
    return _fixtures([(event, 2, 1, 3, fdr)])


def test_dataframe_entrypoint_returns_complete_finite_auditable_rows():
    players = pd.DataFrame(
        [
            _player(1, position="MID"),
            _player(2, team_id=2, position="DEF", price=50),
        ]
    )
    fixtures = _fixtures(
        [
            (5, 1, 2, 2, 4),
            (6, 2, 3, 3, 3),
        ]
    )

    rows = forecast_players_v2(players, fixtures, horizon=2, start_event=5)

    assert len(rows) == 4
    assert set(rows["forecast_version"]) == {FORECAST_VERSION}
    assert {
        "id",
        "player_id",
        "event",
        "gw_offset",
        "ep",
        "expected_minutes",
        "appearance_probability",
        "confidence",
        "reliability",
        "components",
        "component_appearance",
        "component_total",
    }.issubset(rows.columns)
    for _, row in rows.iterrows():
        assert math.isfinite(row["ep"])
        assert math.isfinite(row["expected_minutes"])
        assert 0.0 <= row["appearance_probability"] <= 1.0
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["ep"] == pytest.approx(row["component_total"])
        assert row["ep"] == pytest.approx(row["components"]["total"])
        assert row["reliability"] in {item.value for item in Reliability}


def test_136_minute_cameo_is_not_mistaken_for_a_nailed_player():
    cameo = _player(1, minutes=136, starts=0, sample_matches=10)
    nailed = _player(2, minutes=900, starts=10, sample_matches=10)
    players = pd.DataFrame([cameo, nailed])
    fixtures = _one_fixture()
    priors = build_forecast_priors(players, fixtures=fixtures, start_event=5)

    cameo_minutes = expected_minutes_for_player(cameo, priors)
    nailed_minutes = expected_minutes_for_player(nailed, priors)

    assert nailed_minutes.expected_minutes > 3.0 * cameo_minutes.expected_minutes
    assert nailed_minutes.appearance_probability > cameo_minutes.appearance_probability
    assert nailed_minutes.sixty_probability > cameo_minutes.sixty_probability
    assert nailed_minutes.confidence.score > cameo_minutes.confidence.score


def test_gw1_previous_season_totals_use_team_schedule_not_player_starts():
    cameo = _player(1, minutes=202, starts=2, sample_matches=2)
    regular = _player(2, minutes=2643, starts=29, sample_matches=29)
    schedule_anchor = _player(
        3,
        team_id=2,
        position="GKP",
        minutes=3420,
        starts=38,
        sample_matches=38,
    )
    for player in (cameo, regular, schedule_anchor):
        del player["sample_matches"]
    players = pd.DataFrame([cameo, regular, schedule_anchor])

    priors = build_forecast_priors(players, fixtures=_one_fixture(1), start_event=1)
    cameo_minutes = expected_minutes_for_player(cameo, priors)
    regular_minutes = expected_minutes_for_player(regular, priors)

    assert priors.team_matches_played[1] == 38
    assert cameo_minutes.expected_minutes < 30
    assert regular_minutes.expected_minutes > 2.0 * cameo_minutes.expected_minutes


def test_there_is_no_135_to_136_minute_forecast_cliff():
    below = _player(1, minutes=135, starts=1, sample_matches=5, xg_per90=0.4)
    above = _player(2, minutes=136, starts=1, sample_matches=5, xg_per90=0.4)
    players = pd.DataFrame([below, above])
    fixtures = _one_fixture()
    priors = build_forecast_priors(players, fixtures=fixtures, start_event=5)

    below_forecast = forecast_player_v2(
        below, fixtures, priors, horizon=1, start_event=5
    )
    above_forecast = forecast_player_v2(
        above, fixtures, priors, horizon=1, start_event=5
    )

    assert abs(
        below_forecast.expected_minutes.expected_minutes
        - above_forecast.expected_minutes.expected_minutes
    ) < 0.25
    assert abs(below_forecast.per_gw[0].ep - above_forecast.per_gw[0].ep) < 0.05


def test_new_players_inherit_dynamic_position_and_price_role_priors():
    established = [
        _player(1, price=45, minutes=90, starts=0, xg_per90=0.05),
        _player(2, price=50, minutes=135, starts=1, xg_per90=0.08),
        _player(3, price=65, minutes=450, starts=5, xg_per90=0.20),
        _player(4, price=75, minutes=540, starts=6, xg_per90=0.25),
        _player(5, price=100, minutes=880, starts=10, xg_per90=0.55),
        _player(6, price=110, minutes=900, starts=10, xg_per90=0.65),
    ]
    new_low = _player(7, price=46, minutes=0, starts=0, sample_matches=10)
    new_high = _player(8, price=105, minutes=0, starts=0, sample_matches=10)
    players = pd.DataFrame([*established, new_low, new_high])
    priors = build_forecast_priors(players, start_event=11)

    low_projection = expected_minutes_for_player(new_low, priors)
    high_projection = expected_minutes_for_player(new_high, priors)
    low_prior = priors.for_player(new_low)
    high_prior = priors.for_player(new_high)

    assert low_prior.price_band == "low"
    assert high_prior.price_band == "high"
    assert high_prior.minutes_per_match > low_prior.minutes_per_match
    assert high_prior.xg_per90 > low_prior.xg_per90
    assert high_projection.expected_minutes > low_projection.expected_minutes
    assert high_projection.confidence.reliability is Reliability.LOW
    assert math.isfinite(high_projection.expected_minutes)


def test_empirical_bayes_uses_a_900_minute_rate_prior_continuously():
    background = pd.DataFrame(
        [
            _player(index, minutes=900, starts=10, xg_per90=0.20)
            for index in range(10, 20)
        ]
    )
    priors = build_forecast_priors(
        background,
        start_event=11,
        rate_prior_minutes=DEFAULT_RATE_PRIOR_MINUTES,
    )
    small_sample = _player(1, minutes=90, starts=1, xg_per90=1.0)
    large_sample = _player(2, minutes=1800, starts=20, sample_matches=20, xg_per90=1.0)

    small_rate = shrunk_rates_for_player(small_sample, priors).xg_per90
    large_rate = shrunk_rates_for_player(large_sample, priors).xg_per90
    prior_rate = priors.for_player(small_sample).xg_per90

    assert prior_rate < small_rate < large_rate < 1.0
    nine_hundred = _player(3, minutes=900, starts=10, xg_per90=1.0)
    posterior = shrunk_rates_for_player(nine_hundred, priors).xg_per90
    assert posterior == pytest.approx((1.0 + prior_rate) / 2.0)
    confidence = expected_minutes_for_player(nine_hundred, priors).confidence
    assert confidence.rate_prior_weight == pytest.approx(0.5)


def test_missing_rate_uses_prior_instead_of_becoming_observed_zero():
    players = pd.DataFrame([_player(1), _player(2)])
    priors = build_forecast_priors(players, start_event=11)
    missing = _player(3)
    del missing["expected_goals"]

    rates = shrunk_rates_for_player(missing, priors)

    assert rates.xg_per90 == pytest.approx(priors.for_player(missing).xg_per90)


@pytest.mark.parametrize(
    ("status", "chance", "availability"),
    [
        ("d", 25.0, 0.25),
        ("d", None, 0.50),
        ("i", None, 0.0),
        ("s", 100.0, 0.0),
    ],
)
def test_status_and_chance_scale_minutes_once(
    status: str, chance: float | None, availability: float
):
    available = _player(1)
    players = pd.DataFrame([available])
    fixtures = _one_fixture()
    priors = build_forecast_priors(players, fixtures=fixtures, start_event=5)
    unavailable = _player(2, status=status, chance=chance)

    baseline = forecast_player_v2(
        available, fixtures, priors, horizon=1, start_event=5
    )
    adjusted = forecast_player_v2(
        unavailable, fixtures, priors, horizon=1, start_event=5
    )

    assert adjusted.expected_minutes.availability_probability == availability
    assert adjusted.expected_minutes.expected_minutes == pytest.approx(
        baseline.expected_minutes.expected_minutes * availability
    )
    assert adjusted.per_gw[0].ep == pytest.approx(
        baseline.per_gw[0].ep * availability
    )


def test_blank_and_double_gameweeks_preserve_calendar_shape():
    player = _player(1)
    players = pd.DataFrame([player])
    fixtures = _fixtures(
        [
            (5, 1, 2, 2, 4),
            (5, 3, 1, 3, 2),
            (6, 2, 3, 3, 3),
        ]
    )
    priors = build_forecast_priors(players, fixtures=fixtures, start_event=5)

    forecast = forecast_player_v2(
        player, fixtures, priors, horizon=2, start_event=5
    )
    double, blank = forecast.per_gw

    assert double.event == 5
    assert double.fixtures_count == 2
    assert double.is_dgw is True
    assert double.is_blank is False
    assert double.expected_minutes == pytest.approx(
        2.0 * forecast.expected_minutes.expected_minutes
    )
    assert double.appearance_probability == pytest.approx(
        1.0 - (1.0 - forecast.expected_minutes.appearance_probability) ** 2
    )
    assert blank.event == 6
    assert blank.fixtures_count == 0
    assert blank.is_blank is True
    assert blank.is_dgw is False
    assert blank.expected_minutes == 0.0
    assert blank.appearance_probability == 0.0
    assert blank.ep == 0.0
    assert all(value == 0.0 for value in blank.components.as_dict().values())


def test_fixture_difficulty_and_home_advantage_do_not_change_appearance():
    player = _player(1, xg_per90=0.5, xa_per90=0.3)
    players = pd.DataFrame([player])
    fixtures = _fixtures(
        [
            (5, 1, 2, 1, 5),  # easy home
            (6, 3, 1, 1, 5),  # hard away
        ]
    )
    priors = build_forecast_priors(players, fixtures=fixtures, start_event=5)

    forecast = forecast_player_v2(
        player, fixtures, priors, horizon=2, start_event=5
    )
    easy, hard = forecast.per_gw

    assert easy.components.appearance == pytest.approx(hard.components.appearance)
    assert easy.expected_minutes == pytest.approx(hard.expected_minutes)
    assert easy.components.goals > hard.components.goals
    assert easy.components.assists > hard.components.assists
    assert easy.ep > hard.ep


def test_more_xg_is_monotone_after_shrinkage():
    background = pd.DataFrame([_player(index) for index in range(10, 20)])
    fixtures = _one_fixture()
    priors = build_forecast_priors(background, fixtures=fixtures, start_event=5)
    lower = _player(1, xg_per90=0.20)
    higher = _player(2, xg_per90=0.80)

    low_forecast = forecast_player_v2(
        lower, fixtures, priors, horizon=1, start_event=5
    )
    high_forecast = forecast_player_v2(
        higher, fixtures, priors, horizon=1, start_event=5
    )

    assert high_forecast.rates.xg_per90 > low_forecast.rates.xg_per90
    assert high_forecast.per_gw[0].components.goals > low_forecast.per_gw[0].components.goals
    assert high_forecast.per_gw[0].ep > low_forecast.per_gw[0].ep


def test_defensive_actions_and_goalkeeper_saves_add_2026_27_points():
    background = pd.DataFrame([_player(index) for index in range(10, 20)])
    fixtures = _one_fixture()
    priors = build_forecast_priors(background, fixtures=fixtures, start_event=5)
    low_defender = _player(1, position="DEF", minutes=1800, starts=20, sample_matches=20)
    high_defender = dict(low_defender, id=2, defensive_contribution=240.0)

    low_forecast = forecast_player_v2(
        low_defender, fixtures, priors, horizon=1, start_event=5
    )
    high_forecast = forecast_player_v2(
        high_defender, fixtures, priors, horizon=1, start_event=5
    )

    assert high_forecast.per_gw[0].components.defensive_contribution > 0.0
    assert (
        high_forecast.per_gw[0].components.defensive_contribution
        > low_forecast.per_gw[0].components.defensive_contribution
    )

    goalkeeper = _player(
        3,
        position="GKP",
        minutes=1800,
        starts=20,
        sample_matches=20,
    )
    goalkeeper["saves"] = 80.0
    goalkeeper_priors = build_forecast_priors(
        pd.DataFrame([goalkeeper]), fixtures=fixtures, start_event=5
    )
    goalkeeper_forecast = forecast_player_v2(
        goalkeeper, fixtures, goalkeeper_priors, horizon=1, start_event=5
    )
    assert goalkeeper_forecast.per_gw[0].components.saves > 0.0


def test_raw_official_fixture_columns_are_supported_without_network_io():
    players = pd.DataFrame([_player(1)])
    raw_fixtures = pd.DataFrame(
        [
            {
                "event": 5,
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            }
        ]
    )

    rows = forecast_players_v2(players, raw_fixtures, horizon=1, start_event=5)

    assert rows.iloc[0]["fixtures_count"] == 1
    assert rows.iloc[0]["is_blank"] == False  # noqa: E712 - numpy bool comparison


def test_missing_optional_fdr_is_neutral_instead_of_turning_fixture_into_blank():
    players = pd.DataFrame([_player(1)])
    fixtures = pd.DataFrame([{"event": 5, "home_team": 1, "away_team": 2}])

    rows = forecast_players_v2(players, fixtures, horizon=1, start_event=5)

    assert rows.iloc[0]["fixtures_count"] == 1
    assert rows.iloc[0]["ep"] > 0.0


def test_non_finite_optional_inputs_fail_safe_to_finite_output():
    players = pd.DataFrame(
        [
            {
                "id": 1,
                "team_id": 1,
                "singular_name_short": "unknown",
                "now_cost": float("inf"),
                "minutes": float("inf"),
                "starts": float("-inf"),
                "expected_goals": float("nan"),
                "expected_assists": "not-a-number",
                "bonus": None,
                "clean_sheets": float("inf"),
                "expected_goals_conceded": float("nan"),
                "status": "d",
                "chance_of_playing_next_round": float("inf"),
            }
        ]
    )
    fixtures = _fixtures([(5, 1, 2, float("inf"), 3)])

    rows = forecast_players_v2(players, fixtures, horizon=1, start_event=5)
    row = rows.iloc[0]

    for column in (
        "ep",
        "expected_minutes",
        "appearance_probability",
        "sixty_probability",
        "confidence",
        "component_appearance",
        "component_goals",
        "component_assists",
        "component_clean_sheet",
        "component_bonus",
        "component_defensive_contribution",
        "component_saves",
        "component_goals_conceded",
        "component_total",
    ):
        assert math.isfinite(float(row[column]))
    assert row["expected_minutes"] >= 0.0
    assert 0.0 <= row["appearance_probability"] <= 1.0


def test_empty_pool_has_stable_output_schema():
    rows = forecast_players_v2(
        pd.DataFrame(),
        pd.DataFrame(),
        horizon=2,
        start_event=1,
    )

    assert rows.empty
    assert "ep" in rows.columns
    assert "components" in rows.columns


def test_unused_players_do_not_dilute_the_role_prior_for_observed_starters():
    starters = [
        _player(index, position="DEF", price=50, minutes=180, starts=2, sample_matches=2)
        for index in range(1, 21)
    ]
    reserves = [
        _player(index, position="DEF", price=50, minutes=0, starts=0, sample_matches=2)
        for index in range(21, 61)
    ]
    active_priors = build_forecast_priors(pd.DataFrame(starters), start_event=3)
    mixed_priors = build_forecast_priors(pd.DataFrame([*starters, *reserves]), start_event=3)
    active = expected_minutes_for_player(starters[0], active_priors)
    mixed = expected_minutes_for_player(starters[0], mixed_priors)
    reserve = expected_minutes_for_player(reserves[0], mixed_priors)

    assert mixed.expected_minutes == pytest.approx(active.expected_minutes)
    assert mixed.expected_minutes > 80
    assert mixed.sixty_probability > 0.85
    assert reserve.expected_minutes < 12
    # Predictable early-season role is not proof that attacking output is
    # calibrated: its rate prior still has the unchanged 900-minute strength.
    assert mixed.confidence.rate_prior_weight == pytest.approx(900 / 1080)
    assert mixed.confidence.reliability is not Reliability.HIGH


def test_own_start_and_minutes_evidence_leads_after_three_fixtures():
    starter = _player(1, minutes=270, starts=3, sample_matches=3)
    part_time = _player(2, minutes=120, starts=1, sample_matches=3)
    unused = _player(3, minutes=0, starts=0, sample_matches=3)
    priors = build_forecast_priors(pd.DataFrame([starter, part_time, unused]), start_event=4)

    full = expected_minutes_for_player(starter, priors)
    partial = expected_minutes_for_player(part_time, priors)
    none = expected_minutes_for_player(unused, priors)

    assert full.expected_minutes > 75
    assert full.expected_minutes == pytest.approx((270 + priors.for_player(starter).minutes_per_match) / 4)
    assert 30 < partial.expected_minutes < 50
    assert none.expected_minutes < 15
    assert none.expected_minutes < partial.expected_minutes < full.expected_minutes


def test_one_minute_debut_does_not_trigger_a_starter_prior_cliff():
    background = [
        _player(index, minutes=180, starts=2, sample_matches=2)
        for index in range(1, 21)
    ] + [
        _player(index, minutes=0, starts=0, sample_matches=2)
        for index in range(21, 61)
    ]
    priors = build_forecast_priors(pd.DataFrame(background), start_event=3)
    unused = _player(99, minutes=0, starts=0, sample_matches=2)
    debut = dict(unused, minutes=1)

    before = expected_minutes_for_player(unused, priors)
    after = expected_minutes_for_player(debut, priors)

    assert 0 < after.expected_minutes - before.expected_minutes < 1


@pytest.mark.parametrize("mean", [0.0, 0.01, 0.5, 1.35, 2.0, 5.0, 8.0])
def test_goals_conceded_matches_discrete_poisson_scoring(mean):
    exact_sum = sum(
        (goals // 2) * math.exp(-mean) * mean ** goals / math.factorial(goals)
        for goals in range(90)
    )
    assert _expected_conceded_goal_blocks(mean) == pytest.approx(exact_sum, abs=1e-12)


def test_goals_conceded_applies_before_sixty_minutes_and_scales_availability_once():
    player = _player(1, position="DEF")
    priors = build_forecast_priors(pd.DataFrame([player]), start_event=11)
    minutes = replace(
        expected_minutes_for_player(player, priors),
        expected_minutes=30.0,
        baseline_minutes=30.0,
        appearance_probability=1.0,
        sixty_probability=0.0,
    )
    rates = replace(shrunk_rates_for_player(player, priors), xgc_per90=1.35)
    result = _fixture_components(position="DEF", minutes=minutes, rates=rates, fdr=3, is_home=False)
    half_available = _fixture_components(
        position="DEF",
        minutes=replace(minutes, expected_minutes=15, appearance_probability=0.5, availability_probability=0.5),
        rates=rates, fdr=3, is_home=False,
    )
    assert result.goals_conceded == pytest.approx(-_expected_conceded_goal_blocks(0.45))
    assert result.goals_conceded < 0
    assert half_available.goals_conceded == pytest.approx(result.goals_conceded * 0.5)
    ninety = _fixture_components(
        position="DEF", minutes=replace(minutes, expected_minutes=90, sixty_probability=1),
        rates=rates, fdr=3, is_home=False,
    )
    assert ninety.goals_conceded == pytest.approx(-0.44180137818493744)


def test_official_suspension_expiry_is_applied_to_each_fixture_including_dgw():
    player = dict(_player(1, status="s", chance=0), news="Suspended until 19 Sep", news_added="2026-09-08T12:00:00Z")
    fixtures = _fixtures([(4, 1, 2, 3, 3), (4, 3, 1, 3, 3), (5, 1, 4, 3, 3)])
    fixtures["kickoff_time"] = ["2026-09-12T14:00:00Z", "2026-09-19T14:00:00Z", "2026-09-26T14:00:00Z"]
    priors = build_forecast_priors(pd.DataFrame([player]), fixtures=fixtures, start_event=4)
    forecast = forecast_player_v2(player, fixtures, priors, horizon=2, start_event=4)
    healthy_minutes = expected_minutes_for_player(dict(player, status="a", chance_of_playing_next_round=100), priors)

    assert forecast.per_gw[0].expected_minutes == pytest.approx(healthy_minutes.expected_minutes)
    assert forecast.per_gw[1].expected_minutes == pytest.approx(healthy_minutes.expected_minutes)
    assert forecast.per_gw[0].appearance_probability == pytest.approx(healthy_minutes.appearance_probability)


@pytest.mark.parametrize("status,news", [
    ("i", "Knee injury - Expected back 19 Sep"),
    ("i", "Knee injury - Unknown return date"),
    ("s", "Suspended until 31 Sep"),
    ("s", "Suspended until sometime in September"),
    ("s", "Suspended until 19 Jun"),
])
def test_ambiguous_or_injury_return_dates_never_invent_recovery(status, news):
    player = dict(_player(1, status=status, chance=0), news=news, news_added="2026-09-08T12:00:00Z")
    fixtures = _fixtures([(4, 1, 2, 3, 3), (5, 1, 3, 3, 3)])
    fixtures["kickoff_time"] = ["2026-09-12T14:00:00Z", "2026-09-26T14:00:00Z"]
    priors = build_forecast_priors(pd.DataFrame([player]), fixtures=fixtures, start_event=4)

    forecast = forecast_player_v2(player, fixtures, priors, horizon=2, start_event=4)

    assert [row.expected_minutes for row in forecast.per_gw] == [0, 0]


def test_suspension_date_crosses_calendar_year_without_hallucinating_a_year():
    player = dict(_player(1, status="s", chance=0), news="Suspended until 2 Jan", news_added="2026-12-28T12:00:00Z")
    fixtures = _fixtures([(19, 1, 2, 3, 3), (20, 1, 3, 3, 3)])
    fixtures["kickoff_time"] = ["2026-12-31T14:00:00Z", "2027-01-02T14:00:00Z"]
    priors = build_forecast_priors(pd.DataFrame([player]), fixtures=fixtures, start_event=19)

    forecast = forecast_player_v2(player, fixtures, priors, horizon=2, start_event=19)

    assert forecast.per_gw[0].ep == 0
    assert forecast.per_gw[1].ep > 0


def test_missing_fixture_date_does_not_clear_a_suspension():
    player = dict(_player(1, status="s", chance=0), news="Suspended until 19 Sep", news_added="2026-09-08T12:00:00Z")
    fixtures = _one_fixture(6)
    priors = build_forecast_priors(pd.DataFrame([player]), fixtures=fixtures, start_event=6)

    assert forecast_player_v2(player, fixtures, priors, horizon=1, start_event=6).per_gw[0].ep == 0


def test_previous_season_suspension_news_does_not_clear_current_status():
    player = dict(_player(1, status="s", chance=0), news="Suspended until 19 Sep", news_added="2025-09-08T12:00:00Z")
    fixtures = _one_fixture(4)
    fixtures["kickoff_time"] = ["2026-09-26T14:00:00Z"]
    priors = build_forecast_priors(pd.DataFrame([player]), fixtures=fixtures, start_event=4)

    assert forecast_player_v2(player, fixtures, priors, horizon=1, start_event=4).per_gw[0].ep == 0
