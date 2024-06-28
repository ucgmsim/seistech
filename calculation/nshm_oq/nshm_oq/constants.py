from enum import Enum


class IMLevelOptions(str, Enum):
    nshm = "nshm"
    extended = "ext"


class SAPeriodOptions(str, Enum):
    nshm = "nshm"
    extended = "ext"


NSHM_IM_LEVELS = [
    0.0001,
    0.0002,
    0.0004,
    0.0006,
    0.0008,
    0.001,
    0.002,
    0.004,
    0.006,
    0.008,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.5,
    4,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
]

NSHM_SA_PERIODS = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0
]

EXTENDED_SA_PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]

EXTENDEND_N_IM_LEVELS = 200

# Resource files
GMM_LOGIC_TREE_FNAME = "NZ_NSHM_GMM_LT_final_EE.xml"
BACKARC_POLYGON_FNAME = "backarc.json"


BASE_SOURCE_MODEL_FNAME = "base_source_model.xml"
HAZARD_BASE_JOB_INI_FNAME = "hazard_base_job.ini"
DISAGG_BASE_JOB_INI_FNAME = "disagg_base_job.ini"


DEFAULT_UHS_RPS = [25, 50, 100, 150, 250, 500, 1000, 2500, 5000, 10_000]

# Enums
class DisaggType(Enum):
    TecType = "TRT"
    MagDist = "Mag_Dist"
    MagDistEps = "Mag_Dist_Eps"
