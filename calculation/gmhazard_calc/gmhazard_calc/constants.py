from enum import Enum

from qcore.constants import ExtendedStrEnum


class IMDataType(Enum):
    parametric = "parametric"
    non_parametric = "non_parametric"
    mixed = "mixed"


class SourceToSiteDist(ExtendedStrEnum):
    R_rup = 0, "rrup"
    R_jb = 1, "rjb"
    R_x = 2, "rx"
    R_y = 3, "ry"
    Rtvz = 4, "rtvz"

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


class ERFFileType(ExtendedStrEnum):
    flt_nhm = 0, "flt_nhm"
    ds_erf = 1, "custom_ds_erf"

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


class SourceType(Enum):
    distributed = "ds"
    fault = "flt"


class TectonicType(Enum):
    active_shallow = "active_shallow"
    subduction_interface = "subduction_interface"
    subduction_slab = "subduction_slab"
    volcanic = "volcanic"


class NZTASoilClass(Enum):
    """The soil classes as defined by NZTA 2018"""

    rock = "A"
    soft_or_deep_soil = "D"


class NZSSoilClass(Enum):
    """The soil classes as defined by NZS1170.5"""

    rock = "A"
    weak_rock = "B"
    intermediate_soil = "C"
    soft_or_deep_soil = "D"
    very_soft = "E"

    def __str__(self):
        return str(self.value)


class GMSourceType(Enum):
    """The type of the ground motion source
    for GM selection
    """

    historical = "historical"
    simulations = "simulations"
    mixed = "mixed"


class EventType(Enum):
    """Event types for hypocentre distributions"""

    strike_slip = "strike_slip"
    dip_slip = "dip_slip"
    all = "all"

    @classmethod
    def from_rake(cls, rake: float):
        """Converts a rake value to an event type"""
        if -30 <= rake <= 30 or 150 <= rake <= 210:
            return EventType.strike_slip
        elif 60 <= rake <= 120 or -120 <= rake <= -60:
            return EventType.dip_slip
        else:
            return EventType.all


class HypoMethod(Enum):
    """Hypocentre placement methods"""

    latin_hypercube = "latin_hypercube"
    monte_carlo = "monte_carlo"
    uniform_grid = "uniform_grid"


class GMSType(Enum):
    simulation = "simulation"
    empirical = "empirical"
