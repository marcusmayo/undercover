import json

import pytest

from backend import db


def line(description, amount, kind, **extra):
    return {
        "description": description,
        "amount": amount,
        "kind": kind,
        "quantity": 1,
        "nights": 0,
        "hours": 0,
    } | extra


def payout(client, policy_id, benefit, lines, paid_by_others=0):
    response = client.post(
        f"/api/policies/{policy_id}/payout",
        json={"benefit": benefit, "lines": lines, "paid_by_others": paid_by_others},
    )
    assert response.status_code == 200
    return response.json()


def test_the_labs_example(client):
    result = payout(
        client,
        "POL-2001",
        "baggage",
        [
            line("Hard-shell suitcase", 239.99, "item"),
            line("Three T-shirts", 74.97, "item", quantity=3),
        ],
        100,
    )
    assert result == {
        "calculation": [
            {"step": "Hard-shell suitcase", "amount": 239.99},
            {"step": "Three T-shirts", "amount": 74.97},
            {"step": "Less what others have paid or refunded", "amount": -100},
            {"step": "Less the deductible", "amount": -50.0},
        ],
        "payout": 164.96,
        "needs_sign_off": False,
    }


@pytest.mark.parametrize(
    "case",
    [
        (
            "POL-2001",
            "baggage",
            [line("Watch", 450, "item")],
            0,
            "Watch, limited to $300.00",
            250.0,
        ),
        (
            "POL-2009",
            "baggage",
            [line("Laptop", 1400, "electronics")],
            0,
            "Laptop, limited to $1,200.00",
            1150.0,
        ),
        (
            "POL-2001",
            "baggage",
            [line("Laptop", 1400, "electronics")],
            0,
            "Laptop, limited to $300.00",
            250.0,
        ),
        (
            "POL-2006",
            "baggage",
            [line("Cash", 80, "cash"), line("Cash", 70, "cash")],
            0,
            "Cash, up to $100 in all",
            100.0,
        ),
        (
            "POL-2015",
            "trip_delay",
            [
                line("Hotel", 2500, "hotel", nights=2),
                line("Meals", 600, "meals_without_receipts", hours=30),
            ],
            0,
            "Hotel, limited to $2,100.00",
            1500.0,
        ),
        (
            "POL-2019",
            "trip_cancellation",
            [line("Prepaid tour", 4000, "expense")],
            0,
            "Prepaid tour",
            1400.0,
        ),
        (
            "POL-2007",
            "baggage",
            [line("Items", 2000, "item", quantity=20)],
            0,
            "Items",
            750.0,
        ),
        (
            "POL-2006",
            "trip_delay",
            [line("Taxi", 40, "expense")],
            60,
            "Taxi",
            0.0,
        ),
    ],
)
def test_payout_rules(client, case):
    policy_id, benefit, lines, paid, first_step, expected = case
    result = payout(client, policy_id, benefit, lines, paid)
    assert result["calculation"][0]["step"] == first_step
    assert result["payout"] == expected


def test_meals_without_receipts_per_traveler_per_day(client):
    result = payout(
        client,
        "POL-2015",
        "trip_delay",
        [line("Meals", 600, "meals_without_receipts", hours=30)],
    )
    assert result["calculation"][0] == {
        "step": "Meals, limited to $420.00",
        "amount": 420.0,
    }


def test_the_maximum_and_the_floor_are_shown(client):
    capped = payout(
        client, "POL-2007", "baggage", [line("Items", 2000, "item", quantity=20)]
    )
    assert capped["calculation"][-1] == {
        "step": "Less the amount over the maximum of $750",
        "amount": -1150.0,
    }
    nothing = payout(
        client, "POL-2006", "trip_delay", [line("Taxi", 40, "expense")], 60
    )
    assert nothing["calculation"][-1] == {
        "step": "Nothing is left to pay",
        "amount": 20.0,
    }


def test_sign_off_follows_the_approval_limit(client):
    lines = [line("Hospital", 3750, "expense")]
    assert payout(client, "POL-2006", "emergency_medical", lines)["needs_sign_off"]
    with db.engine().begin() as database:
        company = json.loads(db.setting(database, "company"))
        company["payout_approval_limit"] = 5000
        db.run(
            database,
            "UPDATE settings SET content = :content WHERE name = 'company'",
            content=json.dumps(company),
        )
    assert not payout(client, "POL-2006", "emergency_medical", lines)["needs_sign_off"]


@pytest.mark.parametrize(
    "body",
    [
        {"benefit": "baggage", "lines": [{"description": "Bag", "amount": 10}]},
        {"benefit": "spa_day", "lines": [], "paid_by_others": 0},
        {"benefit": "baggage", "lines": []},
    ],
)
def test_invalid_payout_requests(client, body):
    assert client.post("/api/policies/POL-2001/payout", json=body).status_code == 422


def test_unknown_policy(client):
    response = client.post(
        "/api/policies/POL-9999/payout",
        json={"benefit": "baggage", "lines": [], "paid_by_others": 0},
    )
    assert response.status_code == 404
