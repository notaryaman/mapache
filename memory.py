# memory.py
import json, time
from pathlib import Path
from typing import Any, Dict, Optional

MEM_PATH = Path("data/memory.json")
MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load() -> Dict[str, Any]:
    if MEM_PATH.exists():
        try:
            return json.load(open(MEM_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"profile": {}, "prefs": {}, "facts": {}, "stats": {}}

def _save(mem: Dict[str, Any]) -> None:
    json.dump(mem, open(MEM_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def remember(ns: str, key: str, value: Any) -> None:
    mem = _load()
    if ns not in mem: mem[ns] = {}
    mem[ns][key] = {"value": value, "ts": int(time.time())}
    _save(mem)

def recall(ns: str, key: str, default: Optional[Any]=None) -> Any:
    mem = _load()
    try:
        return mem[ns][key]["value"]
    except Exception:
        return default

def all_ns(ns: str) -> Dict[str, Any]:
    mem = _load()
    out = {}
    for k,v in mem.get(ns, {}).items():
        out[k] = v.get("value")
    return out

def forget(ns: str, key: Optional[str]=None) -> None:
    mem = _load()
    if key is None:
        mem[ns] = {}
    else:
        mem.get(ns, {}).pop(key, None)
    _save(mem)
