"""DSC bolus arrival time detection."""

from osipy.dsc.arrival.base import BaseArrivalDetector
from osipy.dsc.arrival.registry import (
    get_arrival_detector,
    list_arrival_detectors,
    register_arrival_detector,
)

__all__ = [
    "BaseArrivalDetector",
    "get_arrival_detector",
    "list_arrival_detectors",
    "register_arrival_detector",
]
