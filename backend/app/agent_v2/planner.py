import json
from backend.app.agent_v2.state import TaskPlan


def planner_prompt(objective, completed=()):
    return [
        {'role': 'system', 'content': 'Plan verifiable user outcomes, not a sequence of operations. A request to find/read/check inputs and then produce ONE business result is ONE task encompassing the entire objective. Navigation, retrieval, extraction, entry and verification are execution phases inside that task; never make them separate tasks. Use multiple tasks only when the user requests multiple independently useful outputs. Each task must have observable outcome criteria, without invented example amounts, dates or identities. Available capabilities: browser for persisted application outcomes or account lookup, documents for a source-grounded answer, python for dataset computation. verification_capability describes the final outcome, not the first tool used. Preserve conditional/no-op outcomes. Do not invent business action recipes. Never mark a task completed yourself. Completed task IDs/goals are immutable when replanning. JSON schema: ' + json.dumps(TaskPlan.model_json_schema())},
        {'role': 'user', 'content': json.dumps({'original_objective': objective, 'completed_tasks': list(completed)})},
    ]
