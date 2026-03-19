import json
import os
import threading
import uuid

RULES_FILE = "rules.json"
_lock = threading.Lock()

VALID_OPERATORS = {">", "<", "==", "!=", ">=", "<="}
VALID_ACTIONS = {"block", "flag", "allow"}


def _load_rules() -> list:
    if not os.path.exists(RULES_FILE):
        return []
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_rules(rules: list):
    tmp_path = RULES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, RULES_FILE)


def get_rules() -> list:
    return _load_rules()


def add_rule(rule: dict) -> dict:
    if rule.get("operator") not in VALID_OPERATORS:
        raise ValueError(f"Invalid operator: {rule.get('operator')}. Must be one of {VALID_OPERATORS}")
    if rule.get("action") not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {rule.get('action')}. Must be one of {VALID_ACTIONS}")

    rule["id"] = str(uuid.uuid4())
    rule.setdefault("enabled", True)

    with _lock:
        rules = _load_rules()
        rules.append(rule)
        _save_rules(rules)
    return rule


def update_rule(rule_id: str, updated: dict) -> dict:
    with _lock:
        rules = _load_rules()
        for i, r in enumerate(rules):
            if r["id"] == rule_id:
                # Merge: keep existing fields, overwrite with updated ones
                merged = {**r, **updated, "id": rule_id}
                rules[i] = merged
                _save_rules(rules)
                return merged
        raise ValueError(f"Rule {rule_id} not found")


def delete_rule(rule_id: str):
    with _lock:
        rules = _load_rules()
        new_rules = [r for r in rules if r["id"] != rule_id]
        if len(new_rules) == len(rules):
            raise ValueError(f"Rule {rule_id} not found")
        _save_rules(new_rules)


def _evaluate_condition(actual_value, operator: str, rule_value) -> bool:
    """Compare actual_value against rule_value using the given operator."""
    try:
        # Try numeric comparison first
        actual_num = float(actual_value)
        rule_num = float(rule_value)
        if operator == ">":
            return actual_num > rule_num
        elif operator == "<":
            return actual_num < rule_num
        elif operator == ">=":
            return actual_num >= rule_num
        elif operator == "<=":
            return actual_num <= rule_num
        elif operator == "==":
            return actual_num == rule_num
        elif operator == "!=":
            return actual_num != rule_num
    except (ValueError, TypeError):
        # Fall back to string comparison for == and !=
        actual_str = str(actual_value)
        rule_str = str(rule_value)
        if operator == "==":
            return actual_str == rule_str
        elif operator == "!=":
            return actual_str != rule_str
    return False


def apply_rules(transaction: dict) -> dict:
    """Apply all enabled rules to a transaction dict.

    Returns:
        {
            "triggered_rules": [list of triggered rule dicts],
            "action": "block" | "flag" | "allow"
        }
    Action priority: block > flag > allow.
    If no rules trigger, action is "allow".
    """
    rules = _load_rules()
    triggered = []

    for rule in rules:
        if not rule.get("enabled", True):
            continue
        field = rule.get("field", "")
        if field not in transaction:
            continue
        actual_value = transaction[field]
        if _evaluate_condition(actual_value, rule["operator"], rule["value"]):
            triggered.append(rule)

    # Determine final action by priority
    final_action = "allow"
    for tr in triggered:
        action = tr.get("action", "allow")
        if action == "block":
            final_action = "block"
            break
        elif action == "flag" and final_action != "block":
            final_action = "flag"

    return {
        "triggered_rules": triggered,
        "action": final_action,
    }
