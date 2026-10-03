import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SESSION_COOKIE_NAME = "sme_process_session"
DEFAULT_SESSION_IDLE_TTL_SECONDS = 7_200
DEFAULT_SESSION_MAX_COUNT = 100
DEFAULT_PUBLIC_DEMO_CHAT_LIMIT = 20
DEFAULT_PUBLIC_DEMO_CHAT_WINDOW_SECONDS = 600
DEFAULT_PUBLIC_DEMO_ANALYSIS_LIMIT = 3
DEFAULT_PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS = 600
DEFAULT_PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION = 40
DEFAULT_PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION = 5
DEFAULT_RAG_INDEX_PATH = PROJECT_ROOT / ".rag" / "northstar-index.json"
DEFAULT_EVALUATION_REPORT_PATH = PROJECT_ROOT / ".evals" / "latest.json"

# Load the local development environment before resolving immutable runtime settings.
load_dotenv(PROJECT_ROOT / ".env")


def resolve_openai_model(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    return values.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def resolve_openai_embedding_model(
    environment: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environment is None else environment
    return (
        values.get("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL).strip()
        or DEFAULT_OPENAI_EMBEDDING_MODEL
    )


def resolve_openai_eval_model(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    return values.get("OPENAI_EVAL_MODEL", "").strip() or resolve_openai_model(values)


def resolve_positive_int(
    name: str,
    default: int,
    environment: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if environment is None else environment
    raw_value = values.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_optional_positive_int(
    name: str,
    environment: Mapping[str, str] | None = None,
) -> int | None:
    values = os.environ if environment is None else environment
    raw_value = values.get(name, "").strip()
    if not raw_value:
        return None
    return resolve_positive_int(name, 1, values)


def resolve_bool(
    name: str,
    default: bool,
    environment: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if environment is None else environment
    raw_value = values.get(name, "").strip().casefold()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def resolve_project_path(
    name: str,
    default: Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a runtime artifact path, relative to the project root if needed."""
    values = os.environ if environment is None else environment
    raw_value = values.get(name, "").strip()
    if not raw_value:
        return default
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


OPENAI_MODEL = resolve_openai_model()
OPENAI_EMBEDDING_MODEL = resolve_openai_embedding_model()
OPENAI_EVAL_MODEL = resolve_openai_eval_model()
SESSION_COOKIE_NAME = (
    os.getenv("SESSION_COOKIE_NAME", DEFAULT_SESSION_COOKIE_NAME).strip()
    or DEFAULT_SESSION_COOKIE_NAME
)
SESSION_COOKIE_SECURE = resolve_bool("SESSION_COOKIE_SECURE", False)
SESSION_IDLE_TTL_SECONDS = resolve_positive_int(
    "SESSION_IDLE_TTL_SECONDS", DEFAULT_SESSION_IDLE_TTL_SECONDS
)
SESSION_MAX_COUNT = resolve_positive_int("SESSION_MAX_COUNT", DEFAULT_SESSION_MAX_COUNT)
PUBLIC_DEMO_CHAT_LIMIT = resolve_positive_int(
    "PUBLIC_DEMO_CHAT_LIMIT", DEFAULT_PUBLIC_DEMO_CHAT_LIMIT
)
PUBLIC_DEMO_CHAT_WINDOW_SECONDS = resolve_positive_int(
    "PUBLIC_DEMO_CHAT_WINDOW_SECONDS", DEFAULT_PUBLIC_DEMO_CHAT_WINDOW_SECONDS
)
PUBLIC_DEMO_ANALYSIS_LIMIT = resolve_positive_int(
    "PUBLIC_DEMO_ANALYSIS_LIMIT", DEFAULT_PUBLIC_DEMO_ANALYSIS_LIMIT
)
PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS = resolve_positive_int(
    "PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS", DEFAULT_PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS
)
PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION = resolve_positive_int(
    "PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION",
    DEFAULT_PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION,
)
PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION = resolve_positive_int(
    "PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION",
    DEFAULT_PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION,
)
PUBLIC_DEMO_MAX_MODEL_OPERATIONS = resolve_optional_positive_int(
    "PUBLIC_DEMO_MAX_MODEL_OPERATIONS"
)
RAG_INDEX_PATH = resolve_project_path("RAG_INDEX_PATH", DEFAULT_RAG_INDEX_PATH)
EVALUATION_REPORT_PATH = resolve_project_path(
    "EVALUATION_REPORT_PATH", DEFAULT_EVALUATION_REPORT_PATH
)
