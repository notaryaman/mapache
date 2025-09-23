# flows.py
import yaml
from dataclasses import dataclass, field
from typing import List

@dataclass
class Flow:
    id: str
    examples: List[str]
    speak_template: str
    doc_scope: List[str] = field(default_factory=list)  # NEW: optional scoped docs (regex list)

def load_flows(path: str = "flows.yaml") -> List[Flow]:
    data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    flows: List[Flow] = []
    for it in data.get("intents", []):
        flows.append(
            Flow(
                id=it["id"],
                examples=it.get("examples", []),
                speak_template=it["speak_template"],
                doc_scope=it.get("doc_scope", []),  # NEW
            )
        )
    return flows

def render_flow(flow: Flow) -> str:
    # Templates are static prose (no slot-filling here)
    return flow.speak_template
