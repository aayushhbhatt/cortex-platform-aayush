import os
import time
from typing import Any
from dotenv import load_dotenv
from llm.schemas import AllowedAgentName, LLMRequest, LLMResponse, PromptSpec
from reliability.circuit_breaker import CircuitBreaker

_RETRYABLE_LLM_ERROR_TYPES = {
    "rate_limit",
    "timeout",
    "provider_error",
}

load_dotenv()
def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_clamped(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float_clamped(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def get_default_llm_model() -> str:
    return os.getenv("CORTEX_LLM_MODEL", "gpt-4o-mini")


def get_llm_max_attempts() -> int:
    return _env_int_clamped("CORTEX_LLM_MAX_ATTEMPTS", default=3, minimum=1, maximum=5)


def get_llm_base_delay_seconds() -> float:
    return _env_float_clamped("CORTEX_LLM_BASE_DELAY_SECONDS", default=0.25, minimum=0.0, maximum=5.0)


def get_llm_backoff_multiplier() -> float:
    return _env_float_clamped("CORTEX_LLM_BACKOFF_MULTIPLIER", default=2.0, minimum=1.0, maximum=5.0)


def get_llm_circuit_failure_threshold() -> int:
    return _env_int_clamped("CORTEX_LLM_CIRCUIT_FAILURE_THRESHOLD", default=5, minimum=1, maximum=20)


def get_llm_circuit_recovery_seconds() -> float:
    return _env_float_clamped("CORTEX_LLM_CIRCUIT_RECOVERY_SECONDS", default=60.0, minimum=1.0, maximum=600.0)


def is_llm_enabled() -> bool:
    return _env_bool("ENABLE_CORTEX_LLM", default=False)


def _build_llm_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(
        failure_threshold=get_llm_circuit_failure_threshold(),
        recovery_timeout_seconds=get_llm_circuit_recovery_seconds(),
    )


def _llm_circuit_state() -> str:
    return str(_LLM_CIRCUIT_BREAKER.state.value)


def reset_llm_circuit_breaker_for_tests() -> None:
    global _LLM_CIRCUIT_BREAKER
    _LLM_CIRCUIT_BREAKER = _build_llm_circuit_breaker()


def _classify_provider_error(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "rate limit" in text or "429" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(token in text for token in ("authentication", "unauthorized", "401", "api key")):
        return "auth_error"
    return "provider_error"


_LLM_CIRCUIT_BREAKER = _build_llm_circuit_breaker()


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or get_default_llm_model()
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _complete_once(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            return LLMResponse(success=False, error_type="provider_error", message=str(exc))

        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            create_kwargs: dict[str, Any] = {
                "model": request.model or self.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
            }
            if request.response_format == "json":
                create_kwargs["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(**create_kwargs)
            usage = completion.usage.model_dump() if completion.usage else {}

            return LLMResponse(
                success=True,
                content=completion.choices[0].message.content or "",
                model=getattr(completion, "model", None) or request.model,
                usage=usage,
                raw=completion.model_dump(),
            )
        except Exception as exc:
            return LLMResponse(success=False, error_type=_classify_provider_error(exc), message=str(exc))

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not is_llm_enabled():
            return LLMResponse(success=False, error_type="llm_disabled", message="LLM calls are disabled.")
        if not self.is_configured():
            return LLMResponse(success=False, error_type="auth_error", message="OPENAI_API_KEY is not configured.")
        if not _LLM_CIRCUIT_BREAKER.allow_request():
            return LLMResponse(
                success=False,
                error_type="circuit_open",
                message="LLM provider circuit is open; deterministic fallback should be used.",
                raw={"reliability": {"attempts": 0, "max_attempts": get_llm_max_attempts(), "circuit_state": _llm_circuit_state(), "retried": False}},
            )

        attempts = get_llm_max_attempts()
        delay = get_llm_base_delay_seconds()
        backoff = get_llm_backoff_multiplier()
        last_response: LLMResponse | None = None

        for attempt in range(1, attempts + 1):
            response = self._complete_once(request)

            if response.success:
                _LLM_CIRCUIT_BREAKER.record_success()
                response.raw = dict(response.raw or {})
                response.raw["reliability"] = {
                    "attempts": attempt,
                    "max_attempts": attempts,
                    "circuit_state": _llm_circuit_state(),
                    "retried": attempt > 1,
                }
                return response

            last_response = response
            if response.error_type == "auth_error":
                return response

            if response.error_type in _RETRYABLE_LLM_ERROR_TYPES:
                _LLM_CIRCUIT_BREAKER.record_failure()
                if _LLM_CIRCUIT_BREAKER.state.value == "open":
                    return LLMResponse(
                        success=False,
                        error_type="circuit_open",
                        message="LLM provider circuit opened after repeated failures.",
                        raw={
                            "last_error": response.model_dump(),
                            "reliability": {
                                "attempts": attempt,
                                "max_attempts": attempts,
                                "circuit_state": _llm_circuit_state(),
                                "retried": attempt > 1,
                            },
                        },
                    )
                if attempt < attempts:
                    time.sleep(delay)
                    delay *= backoff
                continue

            return response

        if last_response is None:
            return LLMResponse(success=False, error_type="provider_error", message="LLM call failed.")

        last_response.raw = dict(last_response.raw or {})
        last_response.raw["reliability"] = {
            "attempts": attempts,
            "max_attempts": attempts,
            "circuit_state": _llm_circuit_state(),
            "retried": attempts > 1,
        }
        return last_response


def build_json_request(
    *,
    agent: AllowedAgentName,
    prompt: PromptSpec,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 512,
) -> LLMRequest:
    return LLMRequest(
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        agent=agent,
        system_prompt=prompt.system,
        user_prompt=user_prompt,
        model=model or get_default_llm_model(),
        max_tokens=max_tokens,
        response_format="json",
    )
