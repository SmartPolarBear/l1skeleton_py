from enum import Enum, unique


@unique
class CenterType(Enum):
    NON_BRANCH = 1,
    BRANCH = 2,
    BRIDGE = 3,
    REMOVED = 4,
