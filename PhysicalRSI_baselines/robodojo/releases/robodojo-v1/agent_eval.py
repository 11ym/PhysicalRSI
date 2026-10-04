"""Validate the agent-in-the-loop evaluation boundary before XPolicyLab starts."""
import os


def require_agent_key() -> str:
    key = os.environ.get("PHYSICALRSI_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("PHYSICALRSI_AGENT_API_KEY or OPENAI_API_KEY is required")
    return key


if __name__ == "__main__":
    require_agent_key()
    print("agent_key_present")
