import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.prompt_loader import PromptLoadError, load_prompt, render_prompt

def test_load_prompt_all_supported_agents():
    for a in ["supervisor","knowledge","action","research"]:
        assert load_prompt(a).system.strip()

def test_load_prompt_missing_version_raises():
    with pytest.raises(PromptLoadError): load_prompt("supervisor","v9.9.9")

def test_render_prompt_returns_system_text():
    txt=render_prompt("research"); assert isinstance(txt,str) and len(txt)>0 and "Research" in txt
