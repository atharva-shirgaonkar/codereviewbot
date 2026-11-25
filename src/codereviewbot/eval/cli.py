"""Command-line runner for local pull request evaluations."""

import argparse

from ..agent.llm_client import HttpLLMClient
from ..config import get_settings
from .dataset import load_cases
from .judge import LLMJudge
from .runner import run_eval


