"""UnderCover's payout rules for an approved claim.

The rules come from section 10 of the policy wording, the plan limits and sections 6, 7
and 10 of the claims manual.
"""

from math import ceil, inf
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

CASH_LIMIT = 100.0
HOTEL_NIGHT_LIMIT = 350.0

Benefit = Literal[
    "trip_cancellation",
    "trip_interruption",
    "trip_delay",
    "baggage_delay",
    "baggage",
    "passport",
    "emergency_medical",
]

BENEFIT_MAXIMUM = {
    "trip_cancellation": "trip_cancellation_max",
    "trip_interruption": "trip_cancellation_max",
    "trip_delay": "trip_delay_max",
    "baggage_delay": "baggage_delay_max",
    "baggage": "baggage_max",
    "passport": "passport_max",
    "emergency_medical": "emergency_medical_max",
}


class Line(BaseModel):
    """One covered item or expense on a claim."""

    description: str
    amount: float = Field(ge=0)
    quantity: int = Field(ge=1)
    kind: Literal[
        "item", "electronics", "cash", "expense", "hotel", "meals_without_receipts"
    ]
    nights: int = Field(ge=0)
    hours: float = Field(ge=0)


class PayoutRequest(BaseModel):
    """The benefit, the covered lines and what others have paid for the same loss."""

    benefit: Benefit
    lines: list[Line]
    paid_by_others: float = Field(ge=0)


class Step(TypedDict):
    """One line of the calculation."""

    step: str
    amount: float


class Payout(TypedDict):
    """The payout on a claim and the calculation behind it."""

    calculation: list[Step]
    payout: float
    needs_sign_off: bool


def step(description: str, amount: float) -> Step:
    """Make one line of the calculation, rounded to the cent."""
    return {"step": description, "amount": round(amount, 2)}


class Rules:
    """The payout rules for one policy, from the plans and the company settings."""

    def __init__(
        self, policy: dict[str, Any], plans: dict[str, Any], company: dict[str, Any]
    ) -> None:
        self.policy = policy
        self.plan = next(p for p in plans["plans"] if p["plan"] == policy["plan"])
        self.add_ons = {a["add_on"]: a for a in plans["add_ons"]}
        self.deductible_benefits = plans["deductible_applies_to"]
        self.meals = company["meal_allowance_without_receipts"]["amount"]
        self.approval_limit = company["payout_approval_limit"]

    def limit_for(self, line: Line) -> float:
        """Return the most UnderCover pays for one line, or infinity for no limit."""
        travelers = len(self.policy["travelers"])
        match line.kind:
            case "item":
                return self.plan["per_item_limit"] * line.quantity
            case "electronics":
                per_item = (
                    self.add_ons["electronics_plus"]["electronics_per_item_limit"]
                    if "electronics_plus" in self.policy["add_ons"]
                    else self.plan["per_item_limit"]
                )
                return per_item * line.quantity
            case "hotel":
                return HOTEL_NIGHT_LIMIT * ceil(travelers / 2) * line.nights
            case "meals_without_receipts":
                return self.meals * travelers * ceil(line.hours / 24)
            case _:
                return inf

    def eligible_amount(self, line: Line) -> Step:
        """Return the step for one line: the amount claimed, or its limit if lower."""
        limit = self.limit_for(line)
        if line.amount > limit:
            return step(f"{line.description}, limited to ${limit:,.2f}", limit)
        return step(line.description, line.amount)

    def benefit_maximum(self, benefit: Benefit) -> float:
        """Return the plan's maximum for a benefit, capped at the trip cost."""
        maximum = self.plan[BENEFIT_MAXIMUM[benefit]]
        if benefit in ("trip_cancellation", "trip_interruption"):
            return min(maximum, self.policy["trip"]["trip_cost"])
        return maximum

    def work_out(self, request: PayoutRequest) -> Payout:
        """Work out the payout on an approved claim, following section 10."""
        lines = request.lines
        calculation = [
            self.eligible_amount(line) for line in lines if line.kind != "cash"
        ]
        cash = sum(line.amount for line in lines if line.kind == "cash")
        if cash:
            calculation.append(
                step(f"Cash, up to ${CASH_LIMIT:,.0f} in all", min(cash, CASH_LIMIT))
            )
        total = sum(entry["amount"] for entry in calculation)

        if request.paid_by_others:
            calculation.append(
                step("Less what others have paid or refunded", -request.paid_by_others)
            )
            total -= request.paid_by_others

        if request.benefit in self.deductible_benefits and total > 0:
            deductible = min(self.plan["deductible"], total)
            if deductible:
                calculation.append(step("Less the deductible", -deductible))
                total -= deductible

        maximum = self.benefit_maximum(request.benefit)
        if total > maximum:
            calculation.append(
                step(
                    f"Less the amount over the maximum of ${maximum:,.0f}",
                    maximum - total,
                )
            )
            total = maximum

        if total < 0:
            calculation.append(step("Nothing is left to pay", -total))
            total = 0.0

        payout = round(total, 2)
        return {
            "calculation": calculation,
            "payout": payout,
            "needs_sign_off": payout > self.approval_limit,
        }
