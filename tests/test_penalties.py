"""Grid penalties: qualifying order + penalties -> starting grid."""
import pytest

from f1_predict import apply_penalties

Q = list("ABCDEFGHIJ")  # qualifying order, A on pole


def order(penalties, quali=Q):
    grid = apply_penalties(quali, penalties)
    return "".join(sorted(grid, key=grid.get))


def places(d, n):
    return {"driver": d, "kind": "places", "places": n}


def back(d):
    return {"driver": d, "kind": "back"}


def pitlane(d):
    return {"driver": d, "kind": "pitlane"}


def test_no_penalties_keeps_qualifying_order():
    assert order([]) == "ABCDEFGHIJ"


def test_place_penalty_drops_driver_and_others_move_up():
    # B qualified 2nd, drops 2 places -> starts 4th; C and D move up
    assert order([places("B", 2)]) == "ACDBEFGHIJ"


def test_two_place_penalties_applied_in_qualifying_order():
    assert order([places("A", 1), places("B", 1)]) == "CABDEFGHIJ"


def test_drop_larger_than_field_lands_at_the_back_of_place_group():
    assert order([places("C", 50)]) == "ABDEFGHIJC"


def test_back_of_grid_then_pit_lane_each_in_qualifying_order():
    assert order([back("A"), pitlane("B"), back("C")]) == "DEFGHIJACB"


def test_place_penalty_never_ends_up_behind_back_of_grid_drivers():
    assert order([places("I", 5), back("A")]) == "BCDEFGHJIA"


def test_unknown_driver_is_ignored():
    assert order([back("Z")]) == "ABCDEFGHIJ"


def test_zero_place_penalty_changes_nothing():
    assert order([places("D", 0)]) == "ABCDEFGHIJ"


def test_result_is_a_complete_grid():
    grid = apply_penalties(Q, [places("B", 3), back("E"), pitlane("A")])
    assert sorted(grid) == sorted(Q)
    assert sorted(grid.values()) == list(range(1, len(Q) + 1))


def test_las_vegas_2026_piastri_case():
    """Qualified P7 with a 10-place drop in a 22-car field -> started P17."""
    field = [f"d{i}" for i in range(1, 23)]
    assert apply_penalties(field, [places("d7", 10)])["d7"] == 17


@pytest.mark.parametrize("n", [1, 3, 5, 9])
def test_single_drop_lands_exactly_n_places_back(n):
    grid = apply_penalties(Q, [places("A", n)])
    assert grid["A"] == 1 + n
