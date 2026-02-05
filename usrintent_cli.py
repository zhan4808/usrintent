#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_DB_PATH = os.path.join(os.getcwd(), "usrintent.sqlite")
DEFAULT_PROVIDER = os.environ.get("USRINTENT_PROVIDER", "mock")
DEFAULT_ENV_PATH = os.path.join(os.getcwd(), ".usrintent.env")
DEFAULT_DESKTOP_DB = os.path.expanduser(
    os.environ.get(
        "USRINTENT_DESKTOP_DB",
        "~/Library/Application Support/usrintent/usrintent.sqlite",
    )
)
DEFAULT_SYNC_DIR = os.path.expanduser(
    os.environ.get("USRINTENT_SYNC_DIR", "~/.usrintent_sync")
)
DEFAULT_SHARE_DIR = os.path.expanduser(
    os.environ.get("USRINTENT_SHARE_DIR", "~/Library/Application Support/usrintent/share")
)
DEFAULT_SHARE_FILE = "prompt_library.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS intents (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            task_type TEXT,
            constraints_json TEXT,
            done_criteria_json TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            intent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            derived_from TEXT,
            judge_score REAL,
            user_rating REAL,
            output TEXT,
            FOREIGN KEY(intent_id) REFERENCES intents(id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            user_rating REAL,
            user_edits TEXT,
            judge_scores_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(prompt_id) REFERENCES prompts(id)
        );
        """
    )
    conn.commit()


def parse_json_list(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return json.dumps(items)


def generate_prompt(intent: sqlite3.Row) -> str:
    constraints = json.loads(intent["constraints_json"] or "[]")
    done_criteria = json.loads(intent["done_criteria_json"] or "[]")
    parts = [
        "Task:",
        intent["description"],
    ]
    if intent["task_type"]:
        parts.extend(["Task type:", intent["task_type"]])
    if constraints:
        parts.extend(["Constraints:", "- " + "\n- ".join(constraints)])
    if done_criteria:
        parts.extend(["Done criteria:", "- " + "\n- ".join(done_criteria)])
    parts.append("Respond with the best possible solution.")
    return "\n".join(parts)


def judge_output(prompt: str, output: str) -> Dict[str, float]:
    length_score = min(1.0, max(0.0, len(output) / 2000.0))
    return {
        "relevance": 0.7,
        "correctness": 0.6,
        "coherence": 0.7,
        "risk": 0.1,
        "overall": round(0.4 * length_score + 0.6 * 0.7, 3),
    }


def format_row(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    if "constraints_json" in data:
        data["constraints"] = json.loads(data.pop("constraints_json") or "[]")
    if "done_criteria_json" in data:
        data["done_criteria"] = json.loads(data.pop("done_criteria_json") or "[]")
    if "judge_scores_json" in data:
        data["judge_scores"] = json.loads(data.pop("judge_scores_json") or "{}")
    return data


def load_env(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def ensure_watch_path(db_path: str) -> str:
    base_dir = os.path.dirname(db_path) or os.getcwd()
    return os.path.join(base_dir, ".usrintent_watch")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(sync_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(sync_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(sync_dir: str, manifest: Dict[str, Any]) -> None:
    ensure_dir(sync_dir)
    manifest_path = os.path.join(sync_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def average_rating(conn: sqlite3.Connection, prompt_id: str) -> Optional[float]:
    rows = conn.execute(
        "SELECT user_rating FROM feedback WHERE prompt_id = ? AND user_rating IS NOT NULL",
        (prompt_id,),
    ).fetchall()
    if not rows:
        return None
    values = [row["user_rating"] for row in rows]
    return round(sum(values) / len(values), 3)


def call_openai(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY for provider=openai.")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"OpenAI API error: {exc.read().decode('utf-8')}") from exc
    return response["choices"][0]["message"]["content"]


def call_anthropic(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY for provider=anthropic.")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Anthropic API error: {exc.read().decode('utf-8')}") from exc
    return response["content"][0]["text"]


def generate_output(provider: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    if provider == "openai":
        return call_openai(model, prompt, temperature, max_tokens)
    if provider == "anthropic":
        return call_anthropic(model, prompt, temperature, max_tokens)
    return f"[mock output] {prompt.splitlines()[1] if prompt.splitlines() else 'No prompt'}"


def judge_with_model(provider: str, model: str, prompt: str, output: str) -> Dict[str, float]:
    rubric = (
        "Score the output from 0.0 to 1.0 for each category and return JSON only.\n"
        "Categories: relevance, correctness, coherence, risk, overall.\n"
        f"Prompt:\n{prompt}\n\nOutput:\n{output}\n"
    )
    response = generate_output(provider, model, rubric, 0.0, 300)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return judge_output(prompt, output)


def update_intent_timestamp(conn: sqlite3.Connection, intent_id: str) -> None:
    conn.execute(
        "UPDATE intents SET updated_at = ? WHERE id = ?",
        (now_iso(), intent_id),
    )


def cmd_intent_new(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    intent_id = new_id("intent")
    constraints_json = parse_json_list(args.constraints)
    done_json = parse_json_list(args.done_criteria)
    timestamp = now_iso()
    conn.execute(
        """
        INSERT INTO intents (id, description, task_type, constraints_json, done_criteria_json, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            intent_id,
            args.description,
            args.type,
            constraints_json,
            done_json,
            "draft",
            timestamp,
            timestamp,
        ),
    )
    conn.commit()
    print(intent_id)


def cmd_intent_get(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = conn.execute("SELECT * FROM intents WHERE id = ?", (args.intent_id,)).fetchone()
    if not row:
        raise SystemExit(f"Intent not found: {args.intent_id}")
    print(json.dumps(format_row(row), indent=2))


def cmd_intent_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute(
        """
        SELECT id, description, task_type, status, updated_at
        FROM intents
        ORDER BY updated_at DESC
        """
    ).fetchall()
    print(json.dumps([dict(row) for row in rows], indent=2))


def cmd_run(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    intent = conn.execute("SELECT * FROM intents WHERE id = ?", (args.intent_id,)).fetchone()
    if not intent:
        raise SystemExit(f"Intent not found: {args.intent_id}")
    prompt_content = generate_prompt(intent)
    prompt_id = new_id("prompt")
    output = generate_output(args.provider, args.model, prompt_content, args.temperature, args.max_tokens)
    if args.judge_provider and args.judge_model:
        scores = judge_with_model(args.judge_provider, args.judge_model, prompt_content, output)
    else:
        scores = judge_output(prompt_content, output)
    conn.execute(
        """
        INSERT INTO prompts (id, intent_id, content, created_at, derived_from, judge_score, user_rating, output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt_id,
            intent["id"],
            prompt_content,
            now_iso(),
            None,
            scores["overall"],
            None,
            output,
        ),
    )
    update_intent_timestamp(conn, intent["id"])
    conn.commit()
    result = {
        "prompt_id": prompt_id,
        "output": output,
        "judge_scores": scores,
    }
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(output)
    print(json.dumps(result, indent=2))


def cmd_feedback(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (args.prompt_id,)).fetchone()
    if not prompt:
        raise SystemExit(f"Prompt not found: {args.prompt_id}")
    edits_text = None
    if args.edit_file:
        if not os.path.exists(args.edit_file):
            print(f"Warning: edit file not found: {args.edit_file}", file=sys.stderr)
        else:
            with open(args.edit_file, "r", encoding="utf-8") as handle:
                edits_text = handle.read().strip()
    feedback_id = new_id("fb")
    judge_scores_json = json.dumps({})
    conn.execute(
        """
        INSERT INTO feedback (id, prompt_id, user_rating, user_edits, judge_scores_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            feedback_id,
            args.prompt_id,
            args.rating,
            edits_text,
            judge_scores_json,
            now_iso(),
        ),
    )
    if args.rating is not None:
        conn.execute("UPDATE prompts SET user_rating = ? WHERE id = ?", (args.rating, args.prompt_id))
    update_intent_timestamp(conn, prompt["intent_id"])
    conn.commit()
    print(feedback_id)


def cmd_refine(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    prompt = conn.execute(
        "SELECT * FROM prompts WHERE intent_id = ? ORDER BY created_at DESC LIMIT 1",
        (args.intent_id,),
    ).fetchone()
    if not prompt:
        raise SystemExit(f"No prompt versions for intent: {args.intent_id}")
    feedback = conn.execute(
        "SELECT * FROM feedback WHERE prompt_id = ? ORDER BY created_at DESC LIMIT 1",
        (prompt["id"],),
    ).fetchone()
    feedback_note = feedback["user_edits"] if feedback and feedback["user_edits"] else "No user edits provided."
    refined_content = (
        f"{prompt['content']}\n\nRefinement notes:\n- {feedback_note}\n"
        "Update the response to incorporate the notes."
    )
    prompt_id = new_id("prompt")
    output = f"[mock refined output] {prompt['intent_id']}"
    scores = judge_output(refined_content, output)
    conn.execute(
        """
        INSERT INTO prompts (id, intent_id, content, created_at, derived_from, judge_score, user_rating, output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt_id,
            prompt["intent_id"],
            refined_content,
            now_iso(),
            prompt["id"],
            scores["overall"],
            None,
            output,
        ),
    )
    update_intent_timestamp(conn, prompt["intent_id"])
    conn.commit()
    result = {
        "prompt_id": prompt_id,
        "derived_from": prompt["id"],
        "judge_scores": scores,
    }
    print(json.dumps(result, indent=2))


def cmd_history(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute(
        """
        SELECT id, created_at, derived_from, judge_score, user_rating
        FROM prompts
        WHERE intent_id = ?
        ORDER BY created_at ASC
        """,
        (args.intent_id,),
    ).fetchall()
    if not rows:
        print("[]")
        return
    history = [dict(row) for row in rows]
    print(json.dumps(history, indent=2))


def cmd_prompt_get(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = conn.execute("SELECT * FROM prompts WHERE id = ?", (args.prompt_id,)).fetchone()
    if not row:
        raise SystemExit(f"Prompt not found: {args.prompt_id}")
    print(json.dumps(format_row(row), indent=2))


def cmd_prompt_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute(
        """
        SELECT id, created_at, derived_from, judge_score, user_rating
        FROM prompts
        WHERE intent_id = ?
        ORDER BY created_at ASC
        """,
        (args.intent_id,),
    ).fetchall()
    print(json.dumps([dict(row) for row in rows], indent=2))


def cmd_feedback_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute(
        """
        SELECT id, created_at, user_rating
        FROM feedback
        WHERE prompt_id = ?
        ORDER BY created_at ASC
        """,
        (args.prompt_id,),
    ).fetchall()
    print(json.dumps([dict(row) for row in rows], indent=2))


def cmd_export(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    payload: Dict[str, Any] = {
        "exported_at": now_iso(),
        "intents": [],
        "prompts": [],
        "feedback": [],
    }
    if args.intent_id:
        intents = conn.execute("SELECT * FROM intents WHERE id = ?", (args.intent_id,)).fetchall()
    else:
        intents = conn.execute("SELECT * FROM intents").fetchall()
    payload["intents"] = [format_row(row) for row in intents]
    intent_ids = [row["id"] for row in intents]
    if intent_ids:
        placeholders = ",".join("?" for _ in intent_ids)
        prompts = conn.execute(
            f"SELECT * FROM prompts WHERE intent_id IN ({placeholders})",
            intent_ids,
        ).fetchall()
        payload["prompts"] = [format_row(row) for row in prompts]
        prompt_ids = [row["id"] for row in prompts]
        if prompt_ids:
            placeholders = ",".join("?" for _ in prompt_ids)
            feedback = conn.execute(
                f"SELECT * FROM feedback WHERE prompt_id IN ({placeholders})",
                prompt_ids,
            ).fetchall()
            payload["feedback"] = [format_row(row) for row in feedback]
    output_path = args.file or "usrintent_export.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(output_path)


def cmd_import(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    if not os.path.exists(args.file):
        raise SystemExit(f"Import file not found: {args.file}")
    with open(args.file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    intent_map: Dict[str, str] = {}
    for intent in payload.get("intents", []):
        old_id = intent["id"]
        new_id_value = old_id
        exists = conn.execute("SELECT 1 FROM intents WHERE id = ?", (new_id_value,)).fetchone()
        if exists:
            new_id_value = new_id("intent")
        intent_map[old_id] = new_id_value
        conn.execute(
            """
            INSERT INTO intents (id, description, task_type, constraints_json, done_criteria_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id_value,
                intent["description"],
                intent.get("task_type"),
                json.dumps(intent.get("constraints", [])),
                json.dumps(intent.get("done_criteria", [])),
                intent.get("status", "draft"),
                intent.get("created_at", now_iso()),
                intent.get("updated_at", now_iso()),
            ),
        )
    prompt_map: Dict[str, str] = {}
    for prompt in payload.get("prompts", []):
        old_id = prompt["id"]
        new_id_value = old_id
        exists = conn.execute("SELECT 1 FROM prompts WHERE id = ?", (new_id_value,)).fetchone()
        if exists:
            new_id_value = new_id("prompt")
        prompt_map[old_id] = new_id_value
        intent_id = intent_map.get(prompt["intent_id"], prompt["intent_id"])
        conn.execute(
            """
            INSERT INTO prompts (id, intent_id, content, created_at, derived_from, judge_score, user_rating, output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id_value,
                intent_id,
                prompt["content"],
                prompt.get("created_at", now_iso()),
                prompt.get("derived_from"),
                prompt.get("judge_score"),
                prompt.get("user_rating"),
                prompt.get("output"),
            ),
        )
    for fb in payload.get("feedback", []):
        old_prompt_id = fb["prompt_id"]
        prompt_id = prompt_map.get(old_prompt_id, old_prompt_id)
        fb_id = fb["id"]
        exists = conn.execute("SELECT 1 FROM feedback WHERE id = ?", (fb_id,)).fetchone()
        if exists:
            fb_id = new_id("fb")
        conn.execute(
            """
            INSERT INTO feedback (id, prompt_id, user_rating, user_edits, judge_scores_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                fb_id,
                prompt_id,
                fb.get("user_rating"),
                fb.get("user_edits"),
                json.dumps(fb.get("judge_scores", {})),
                fb.get("created_at", now_iso()),
            ),
        )
    conn.commit()
    print("ok")


def cmd_watch(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    watch_path = ensure_watch_path(args.db)
    if args.action == "start":
        with open(watch_path, "w", encoding="utf-8") as handle:
            handle.write(now_iso())
        print("watch enabled")
    elif args.action == "stop":
        if os.path.exists(watch_path):
            os.remove(watch_path)
        print("watch disabled")
    else:
        status = "enabled" if os.path.exists(watch_path) else "disabled"
        print(f"watch {status}")


def cmd_sync(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    source_db = args.db
    target_db = args.desktop_db
    sync_dir = args.sync_dir
    ensure_dir(os.path.dirname(target_db))
    ensure_dir(sync_dir)
    backups_dir = os.path.join(sync_dir, "backups")
    ensure_dir(backups_dir)

    if args.direction == "pull":
        source_db, target_db = target_db, source_db

    source_hash = file_sha256(source_db)
    target_hash = file_sha256(target_db)

    if args.direction == "status":
        print(
            json.dumps(
                {
                    "cli_db": args.db,
                    "desktop_db": args.desktop_db,
                    "cli_hash": file_sha256(args.db),
                    "desktop_hash": file_sha256(args.desktop_db),
                },
                indent=2,
            )
        )
        return

    if args.interactive:
        print("usrintent sync")
        print("")
        print("Select direction:")
        print("1) CLI -> Desktop")
        print("2) Desktop -> CLI")
        choice = input("Enter choice [1-2]: ").strip()
        if choice == "2":
            source_db, target_db = target_db, source_db
            args.direction = "pull"
        elif choice != "1":
            raise SystemExit("Invalid choice.")
        source_hash = file_sha256(source_db)
        target_hash = file_sha256(target_db)
        print("")
        print("Planned sync:")
        print(f"- source: {source_db}")
        print(f"- target: {target_db}")
        print(f"- source hash: {source_hash}")
        print(f"- target hash: {target_hash}")
        confirm = input("Proceed? [y/N]: ").strip().lower()
        if confirm != "y":
            print("sync cancelled")
            return

    if not os.path.exists(source_db):
        raise SystemExit(f"Source database not found: {source_db}")

    if os.path.exists(target_db):
        backup_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.sqlite"
        backup_path = os.path.join(backups_dir, backup_name)
        with open(target_db, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())

    with open(source_db, "rb") as src, open(target_db, "wb") as dst:
        dst.write(src.read())

    manifest = load_manifest(sync_dir)
    manifest["last_sync"] = now_iso()
    manifest["direction"] = args.direction
    manifest["source_db"] = source_db
    manifest["target_db"] = target_db
    manifest["source_hash"] = source_hash
    manifest["target_hash"] = target_hash
    save_manifest(sync_dir, manifest)

    print(
        json.dumps(
            {
                "direction": args.direction,
                "source_db": source_db,
                "target_db": target_db,
                "source_hash": source_hash,
                "target_hash": target_hash,
            },
            indent=2,
        )
    )


def build_prompt_library(conn: sqlite3.Connection) -> Dict[str, Any]:
    prompts = conn.execute(
        """
        SELECT p.id, p.intent_id, p.content, p.created_at, p.derived_from, p.judge_score, p.user_rating, p.output,
               i.description AS intent_description
        FROM prompts p
        JOIN intents i ON i.id = p.intent_id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    entries = []
    for row in prompts:
        avg_rating = average_rating(conn, row["id"])
        rating = row["user_rating"] if row["user_rating"] is not None else avg_rating
        entries.append(
            {
                "prompt_id": row["id"],
                "intent_id": row["intent_id"],
                "intent_description": row["intent_description"],
                "content": row["content"],
                "created_at": row["created_at"],
                "derived_from": row["derived_from"],
                "judge_score": row["judge_score"],
                "user_rating": rating,
            }
        )
    return {
        "exported_at": now_iso(),
        "count": len(entries),
        "prompts": entries,
    }


def cmd_share(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    ensure_dir(args.share_dir)
    share_path = os.path.join(args.share_dir, DEFAULT_SHARE_FILE)
    if args.action == "push":
        payload = build_prompt_library(conn)
        with open(share_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(share_path)
        return
    if args.action == "pull":
        if not os.path.exists(share_path):
            raise SystemExit(f"Share file not found: {share_path}")
        with open(share_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        imported = 0
        for item in payload.get("prompts", []):
            prompt_id = item["prompt_id"]
            exists = conn.execute("SELECT 1 FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            if exists:
                continue
            intent_id = item["intent_id"]
            intent_exists = conn.execute("SELECT 1 FROM intents WHERE id = ?", (intent_id,)).fetchone()
            if not intent_exists:
                conn.execute(
                    """
                    INSERT INTO intents (id, description, task_type, constraints_json, done_criteria_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent_id,
                        item.get("intent_description", "Imported intent"),
                        None,
                        json.dumps([]),
                        json.dumps([]),
                        "imported",
                        now_iso(),
                        now_iso(),
                    ),
                )
            conn.execute(
                """
                INSERT INTO prompts (id, intent_id, content, created_at, derived_from, judge_score, user_rating, output)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_id,
                    intent_id,
                    item["content"],
                    item.get("created_at", now_iso()),
                    item.get("derived_from"),
                    item.get("judge_score"),
                    item.get("user_rating"),
                    None,
                ),
            )
            imported += 1
        conn.commit()
        print(json.dumps({"imported": imported}, indent=2))
        return
    status = {
        "share_dir": args.share_dir,
        "share_file": share_path,
        "exists": os.path.exists(share_path),
    }
    if status["exists"]:
        status["updated_at"] = datetime.fromtimestamp(os.path.getmtime(share_path)).isoformat()
    print(json.dumps(status, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usrintent", description="Intent refinement CLI")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    intent_parser = subparsers.add_parser("intent", help="Intent commands")
    intent_sub = intent_parser.add_subparsers(dest="intent_cmd", required=True)
    intent_new = intent_sub.add_parser("new", help="Create a new intent")
    intent_new.add_argument("description")
    intent_new.add_argument("--type", dest="type")
    intent_new.add_argument("--constraints")
    intent_new.add_argument("--done-criteria")
    intent_new.set_defaults(func=cmd_intent_new)

    intent_get = intent_sub.add_parser("get", help="Get an intent")
    intent_get.add_argument("intent_id")
    intent_get.set_defaults(func=cmd_intent_get)

    intent_list = intent_sub.add_parser("list", help="List intents")
    intent_list.set_defaults(func=cmd_intent_list)

    run_parser = subparsers.add_parser("run", help="Run a prompt for an intent")
    run_parser.add_argument("intent_id")
    run_parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["mock", "openai", "anthropic"])
    run_parser.add_argument("--model", default="gpt-4o-mini")
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--max-tokens", type=int, default=800)
    run_parser.add_argument("--judge-provider", choices=["mock", "openai", "anthropic"])
    run_parser.add_argument("--judge-model")
    run_parser.add_argument("--output-file")
    run_parser.set_defaults(func=cmd_run)

    feedback_parser = subparsers.add_parser("feedback", help="Submit feedback for a prompt")
    feedback_parser.add_argument("prompt_id")
    feedback_parser.add_argument("--rating", type=float)
    feedback_parser.add_argument("--edit-file")
    feedback_parser.set_defaults(func=cmd_feedback)

    refine_parser = subparsers.add_parser("refine", help="Refine the latest prompt for an intent")
    refine_parser.add_argument("intent_id")
    refine_parser.set_defaults(func=cmd_refine)

    history_parser = subparsers.add_parser("history", help="Show prompt history for an intent")
    history_parser.add_argument("intent_id")
    history_parser.set_defaults(func=cmd_history)

    prompt_parser = subparsers.add_parser("prompt", help="Prompt commands")
    prompt_sub = prompt_parser.add_subparsers(dest="prompt_cmd", required=True)
    prompt_get = prompt_sub.add_parser("get", help="Get a prompt")
    prompt_get.add_argument("prompt_id")
    prompt_get.set_defaults(func=cmd_prompt_get)

    prompt_list = prompt_sub.add_parser("list", help="List prompts for an intent")
    prompt_list.add_argument("intent_id")
    prompt_list.set_defaults(func=cmd_prompt_list)

    feedback_list = subparsers.add_parser("feedback-list", help="List feedback for a prompt")
    feedback_list.add_argument("prompt_id")
    feedback_list.set_defaults(func=cmd_feedback_list)

    export_parser = subparsers.add_parser("export", help="Export intents, prompts, feedback to JSON")
    export_parser.add_argument("--intent-id")
    export_parser.add_argument("--file")
    export_parser.set_defaults(func=cmd_export)

    import_parser = subparsers.add_parser("import", help="Import intents, prompts, feedback from JSON")
    import_parser.add_argument("file")
    import_parser.set_defaults(func=cmd_import)

    watch_parser = subparsers.add_parser("watch", help="Enable/disable watch reminders")
    watch_parser.add_argument("action", choices=["start", "stop", "status"])
    watch_parser.set_defaults(func=cmd_watch)

    sync_parser = subparsers.add_parser("sync", help="Sync CLI and desktop databases")
    sync_parser.add_argument("direction", choices=["push", "pull", "status"])
    sync_parser.add_argument("--desktop-db", default=DEFAULT_DESKTOP_DB)
    sync_parser.add_argument("--sync-dir", default=DEFAULT_SYNC_DIR)
    sync_parser.add_argument("--interactive", action="store_true")
    sync_parser.set_defaults(func=cmd_sync)

    share_parser = subparsers.add_parser("share", help="Share prompt library between CLI and desktop")
    share_parser.add_argument("action", choices=["push", "pull", "status"])
    share_parser.add_argument("--share-dir", default=DEFAULT_SHARE_DIR)
    share_parser.set_defaults(func=cmd_share)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_env(DEFAULT_ENV_PATH)
    conn = connect_db(args.db)
    init_db(conn)
    args.func(conn, args)


if __name__ == "__main__":
    main()
