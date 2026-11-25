"""One-to-one finding matching and evaluation metrics."""

from dataclasses import dataclass
from statistics import fmean
from typing import Protocol, Sequence

from ..models import ReviewComment
from .dataset import GroundTruthIssue


