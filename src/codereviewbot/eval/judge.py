"""LLM-backed and scripted judges for finding comparisons."""

import json
import logging
from typing import Protocol

from ..agent.types import LLMClient
