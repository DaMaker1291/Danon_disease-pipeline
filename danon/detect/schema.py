"""The data contract: every observation is validated before it is believed.

Phase 1 is a research-only engine over SYNTHETIC observations, and the
contract it serves is the one the mission asked for: a timestamp, a
measurement type, units, and provenance. A stream that cannot state those
four things is malformed and is DROPPED, not corrected.

GROUND TRUTH IS NOT OBSERVABLE DATA. The generator keeps the planted change
in a separate :class:`GroundTruth` book; the detector consumes an
:class:`ObservationStream` which is structurally INCAPABLE of carrying it --
the ground truth lives on an object the detector never receives. Nothing
needs to be trusted about that separation at review time: the leakage tests
demonstrate it by attempting to detect a planted change from the stream the
detector is allowed to see when the ground truth has been deleted.

All timestamps are integers: the tiniest quantity the engine can be told.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Units(str, Enum):
    """The units an observation carries, per measurement type."""

    DIAMETER_MM = "mm"
    AREA_MM2 = "mm2"
    VOLUME_MM3 = "volume_mm3"
    RATIO = "ratio"
    COUNTS = "counts"
    GENERIC = "generic"


class MeasurementKind(str, Enum):
    """Which slice of the state an observation reads."""

    RADIUS = "radius"
    AREA = "area"
    VOLUME = "volume"
    GROWTH_RATE = "growth_rate"
    SIGNAL = "signal"


class Provenance(BaseModel):
    """Where a number came from -- every observation states this."""

    model_config = ConfigDict(frozen=True)

    source: str
    generated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    seed: Optional[int] = None
    scenario_kind: str = ""
    config_fingerprint: str = ""


class ETMeasurement(BaseModel):
    """A single validated observation."""

    model_config = ConfigDict(frozen=True)

    day: int = Field(ge=0, description="Timestamp: tiniest quantity the engine can be told.")
    kind: MeasurementKind
    units: Units
    value: float
    sigma: float = Field(gt=0.0, description="Declared measurement uncertainty.")

    @field_validator("sigma")
    @classmethod
    def _finite_sigma(cls, s: float) -> float:
        if not np.isfinite(s):
            raise ValueError("sigma must be finite")
        return float(s)
    provenance: Provenance

    @field_validator("value")
    @classmethod
    def _finite_value(cls, v: float) -> float:
        # Corrupted readings (NaN, inf) are rejected at construction, never
        # admitted and patched downstream: a measurement that cannot state
        # its value is malformed, not uncertain.
        if not np.isfinite(v):
            raise ValueError("observation value must be finite")
        return float(v)


class ObservationStream(BaseModel):
    """A validated observation series for ONE scenario case.

    This is the ONLY data object the detector receives. It carries no label,
    no change day, no scenario kind string beyond the provenance tag -- and
    even the provenance is attachable metadata about the channel, not a
    hint about where the change is.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    observations: Tuple[ETMeasurement, ...] = Field(min_length=2)
    provenance: Provenance

    @field_validator("observations")
    @classmethod
    def _ascending_unique_days(cls, obs: Tuple[ETMeasurement, ...]) -> Tuple[ETMeasurement, ...]:
        days = [o.day for o in obs]
        if any(b <= a for a, b in zip(days, days[1:])):
            raise ValueError("observation days must be strictly ascending")
        return obs

    @property
    def days(self) -> np.ndarray:
        return np.array([o.day for o in self.observations], dtype=float)

    @property
    def values(self) -> np.ndarray:
        return np.array([o.value for o in self.observations], dtype=float)

    @property
    def sigmas(self) -> np.ndarray:
        return np.array([o.sigma for o in self.observations], dtype=float)


class GroundTruth(BaseModel):
    """What actually happened in a synthetic case -- NOT given to the detector.

    Sealed by construction: the generator hands an ObservationStream to the
    experiment, and this book to a LATER stage. The detector never imports
    it. The evaluator receives both and measures the detector against the
    book.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    planted_change_day: Optional[int] = None
    planted_direction: str = ""  # "growth" | "shrink" | ""
    planted_magnitude: float = 0.0
    is_negative_control: bool = False
    missing_days: Tuple[int, ...] = ()
    noise_sigma_used: float = 0.0
    scenario_kind: str = ""

    @field_validator("planted_direction")
    @classmethod
    def _direction(cls, v: str) -> str:
        if v not in ("growth", "shrink", ""):
            raise ValueError("planted_direction must be growth|shrink|''")
        return v


class ScenarioKind(str, Enum):
    SINGLE_CHANGE = "single_change"
    NEGATIVE_CONTROL = "negative_control"


class ScenarioCase(BaseModel):
    """One generated case, split at birth into observable and truth."""

    model_config = ConfigDict(frozen=True)

    stream: ObservationStream
    truth: GroundTruth

    @property
    def case_id(self) -> str:
        return self.stream.case_id


class Scenario(BaseModel):
    """A full generated scenario: many cases, truth kept alongside them."""

    model_config = ConfigDict(frozen=True)

    name: str
    seed: int
    cases: Tuple[ScenarioCase, ...] = Field(min_length=1)

    @property
    def n_cases(self) -> int:
        return len(self.cases)


def ground_truth_summary(scenario: Scenario) -> str:
    """One line about what was planted -- for the report, never the detector."""
    n_change = sum(1 for c in scenario.cases if not c.truth.is_negative_control)
    n_neg = scenario.n_cases - n_change
    return f"{scenario.n_cases} cases: {n_change} with a planted change, {n_neg} negative controls"


class GenomicsSchema(BaseModel):
    """Schema self-description for the record's audit trail."""

    schema_name: str = "evotwin.detect.observations.v1"
    units_vocabulary: List[str] = Field(
        default_factory=lambda: [u.value for u in Units]
    )
    separation_rule: str = (
        "ground truth lives in a GroundTruth book the detector never receives; "
        "leakage is demonstrated structurally, by the no-label tests"
    )


class ETValues(BaseModel):
    """Array container for vectorised downstream maths (kept minimal)."""

    days: np.ndarray
    values: np.ndarray
    sigmas: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)
