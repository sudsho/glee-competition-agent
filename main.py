"""GLEE competition runner for agent `sudsho`.

Rule-based, zero-LLM-cost strategies for all three families. Every move
is logged to logs/moves.jsonl (and game ids to logs/game_ids.txt) so we
can fetch outcomes later, tune parameters, and write the competition
paper's Agent Behavior Analysis from real data.
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glee_sdk import GleeClient, CompetitionClosedError

from barg import bargaining_strategy
from nego import negotiation_strategy
from pers import persuasion_strategy
from safe import safe_action

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path(os.environ.get("GLEE_LOG_DIR") or (ROOT / "logs"))
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS / "runner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("sudsho")

STRATS = {
    "bargaining": bargaining_strategy,
    "negotiation": negotiation_strategy,
    "persuasion": persuasion_strategy,
}

_seen_ids = set()
_log_lock = threading.Lock()


def _record(entry: dict) -> None:
    try:
        line = json.dumps(entry, default=str) + "\n"
        with _log_lock, open(LOGS / "moves.jsonl", "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def _record_game_id(game_id: str) -> None:
    if game_id in _seen_ids:
        return
    _seen_ids.add(game_id)
    try:
        with _log_lock, open(LOGS / "game_ids.txt", "a", encoding="utf-8") as f:
            f.write(game_id + "\n")
    except OSError:
        pass


def strategy(game: dict) -> dict:
    t0 = time.time()
    family = game.get("game_family", "?")
    try:
        action = STRATS[family](game)
    except Exception:
        log.exception("strategy error in %s game %s; playing safe",
                      family, game.get("game_id"))
        action = safe_action(game)

    msg = action.get("message")
    if isinstance(msg, str) and len(msg) > 1900:
        action["message"] = msg[:1900]

    st = game.get("game_state") or {}
    seller_msg = None
    if family == "persuasion" and st.get("seller_message") is not None:
        seller_msg = str(st.get("seller_message"))[:200]

    _record_game_id(str(game.get("game_id")))
    _record({
        "seller_msg": seller_msg,
        "ts": time.time(),
        "game_id": game.get("game_id"),
        "family": family,
        "your_player": game.get("your_player"),
        "phase": game.get("phase"),
        "round": st.get("round"),
        "opponent": game.get("opponent"),
        "action_type": (game.get("valid_actions") or {}).get("type"),
        "action": action,
        # Full decision-relevant payload: enables exact policy replay for
        # the behavior audit. The server's prose prompt is excluded.
        "game_state": st,
        "valid_actions": game.get("valid_actions"),
        "ms": int((time.time() - t0) * 1000),
    })
    return action


def api_key() -> str:
    key = os.environ.get("GLEE_API_KEY", "").strip()
    if not key:
        key = (ROOT / "key.txt").read_text(encoding="utf-8").strip()
    return key


def main() -> None:
    client = GleeClient(api_key=api_key())
    log.info("stats at start: %s", client.stats())
    while True:
        try:
            client.run(strategy, concurrency=8, poll_interval=3.0)
            log.info("run() returned; restarting in 5s")
            time.sleep(5)
        except CompetitionClosedError:
            log.info("competition closed; exiting")
            break
        except KeyboardInterrupt:
            log.info("interrupted; exiting")
            break
        except Exception:
            log.exception("run loop crashed; restarting in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
