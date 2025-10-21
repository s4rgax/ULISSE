from enum import Enum


class TemporalMode(Enum):
    SINGLE = "single"
    TIMESERIES = "timeseries"

class FusionType(Enum):
    MIDDLE = "middle"

class FusionTechnique(Enum):
    CONCATENATION = "concat"
    DIFF = "dif"
    SUM = "sum"


class PeftMode(Enum):
    LORA = "lora"
    HRA = "hra"
    DORA = "dora"
    QLORA = "qlora"
    QDORA = "qdora"