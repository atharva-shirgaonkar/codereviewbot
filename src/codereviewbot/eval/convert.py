"""Generic CSV/JSON record conversion to the normalized case format."""

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .dataset import EvalCase
