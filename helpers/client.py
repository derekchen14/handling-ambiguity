"""Unified LLM client wrapping 5 providers (6 API paths) for experiment calls."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

# Provider → model-level → model ID mapping (Section C.2)
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    'anthropic': {
        'low': 'claude-haiku-4-5-20251001',
        'medium': 'claude-sonnet-4-6',
        'high': 'claude-opus-4-6',
    },
    'google': {
        'low': 'gemini-3-flash-preview',
        'medium': 'gemini-3-pro-preview',
        'high': 'gemini-3-pro-preview',  # thinking mode differentiates
    },
    'openai': {
        'low': 'gpt-5-nano',
        'medium': 'gpt-5-mini',
        'high': 'gpt-5.2',
    },
    'qwen': {
        'low': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
        'medium': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
        'high': 'Qwen/Qwen3-235B-A22B-Thinking-2507',
    },
    'deepseek': {
        'low': 'deepseek-chat',           # DeepSeek-V3 (temp=0.1 to differentiate from medium)
        'medium': 'deepseek-chat',        # DeepSeek-V3
        'high': 'deepseek-reasoner',      # DeepSeek-R1
    },
    # ── OpenRouter (any model via OpenAI-compatible API) ───────────
    'openrouter': {
        'high': 'google/gemini-3.1-pro-preview',
    },
    # ── Gemma (open model via Google API) ──────────────────────────
    # Uses the same GOOGLE_API_KEY and google-genai SDK as Gemini, but
    # the API surface is restricted — see _call_gemma docstring.
    # Conceptually Google's low-tier model, but uses separate API path.
    # No thinking modes available.
    'gemma': {
        'low': 'gemma-3-27b-it',
    },
}


# ── Per-provider rate limits (RPM) ────────────────────────────────
# Together.AI Tier 1 = 600 RPM.  Others are generous enough to skip.
PROVIDER_RPM: dict[str, int] = {
    'google': 20,     # Gemini 3 Pro: 25 RPM hard limit, leave margin
    'gemma': 20,      # Shares Google API quota; keep conservative
    'qwen': 600,
}


class _TokenBucketLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm      # seconds between requests
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval


class UnifiedLLMClient:
    """Wraps 6 LLM providers behind a single interface."""

    MAX_RETRIES = 3
    BACKOFF_BASE_S = 0.5

    def __init__(self):
        self._anthropic_client = None
        self._google_client = None
        self._openai_client = None
        self._openrouter_client = None
        self._rate_limiters: dict[str, _TokenBucketLimiter] = {
            provider: _TokenBucketLimiter(rpm)
            for provider, rpm in PROVIDER_RPM.items()
        }

    # ── Lazy client init ──────────────────────────────────────────

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.environ['ANTHROPIC_API_KEY'],
            )
        return self._anthropic_client

    @property
    def google_client(self):
        if self._google_client is None:
            from google import genai
            self._google_client = genai.Client(
                api_key=os.environ['GOOGLE_API_KEY'],
            )
        return self._google_client

    @property
    def openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
            )
        return self._openai_client

    @property
    def openrouter_client(self):
        if self._openrouter_client is None:
            import openai
            self._openrouter_client = openai.OpenAI(
                api_key=os.environ['OPEN_ROUTER_API_KEY'],
                base_url='https://openrouter.ai/api/v1',
            )
        return self._openrouter_client

    # ── Public API ────────────────────────────────────────────────

    def call_flow_detection(
        self,
        config: dict,
        system_prompt: str,
        messages: list[dict],
    ) -> dict:
        """Call an LLM for flow detection (text-in, JSON-out).

        Returns:
            {detected_flow, raw_response, latency_ms, input_tokens, output_tokens}
        """
        provider = config['provider']
        self._rate_limit(provider)
        dispatch = {
            'anthropic': self._call_anthropic,
            'google': self._call_google,
            'gemma': self._call_gemma,
            'openai': self._call_openai,
            'openrouter': self._call_openrouter,
            'qwen': self._call_qwen,
            'deepseek': self._call_deepseek,
        }
        fn = dispatch[provider]

        t0 = time.perf_counter()
        result = self._with_retries(fn, config, system_prompt, messages)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        result['latency_ms'] = latency_ms
        parsed = self._parse_flows(result['raw_response'])
        result['detected_flows'] = parsed['flows']
        result['reasoning'] = parsed['reasoning']
        return result

    def call_tool_use(
        self,
        config: dict,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        """Call an LLM for tool-use flow detection.

        Returns:
            {tool_called, tool_args, raw_response, latency_ms, input_tokens, output_tokens}
        """
        provider = config['provider']
        self._rate_limit(provider)
        dispatch = {
            'anthropic': self._call_anthropic,
            'google': self._call_google,
            'gemma': self._call_gemma,
            'openai': self._call_openai,
            'openrouter': self._call_openrouter,
            'qwen': self._call_qwen,
            'deepseek': self._call_deepseek,
        }
        fn = dispatch[provider]

        t0 = time.perf_counter()
        result = self._with_retries(fn, config, system_prompt, messages, tools=tools)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        result['latency_ms'] = latency_ms
        return result

    # ── Rate limiting ──────────────────────────────────────────────

    def _rate_limit(self, provider: str):
        limiter = self._rate_limiters.get(provider)
        if limiter:
            limiter.wait()

    # ── Retry wrapper ─────────────────────────────────────────────

    def _with_retries(self, fn, *args, **kwargs) -> dict:
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                err_str = f'{type(e).__name__}: {e}'.lower()

                # Daily quota exhausted — no point retrying
                if 'per_day' in err_str or 'per_model_per_day' in err_str:
                    log.error('Daily quota exhausted — aborting (no retry)')
                    raise

                is_retryable = any(kw in err_str for kw in (
                    'ratelimit', 'rate_limit', 'rate limit', 'resource_exhausted',
                    'timeout', 'internal', 'server', '429', '500', '503',
                ))
                if not is_retryable:
                    raise
                if attempt < self.MAX_RETRIES:
                    delay = max(self.BACKOFF_BASE_S * (2 ** attempt), 5.0 if '429' in err_str else 0.5)
                    log.warning('Retry %d/%d after %s: %s', attempt + 1, self.MAX_RETRIES, type(e).__name__, e)
                    time.sleep(delay)
        raise last_error

    # ── Provider implementations ──────────────────────────────────

    def _call_anthropic(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        import anthropic

        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)

        kwargs: dict[str, Any] = {
            'model': model_id,
            'max_tokens': 1024,
            'system': system_prompt,
            'messages': messages,
        }

        if temperature > 0:
            kwargs['temperature'] = temperature

        if tools:
            kwargs['tools'] = [
                {k: v for k, v in t.items() if k in ('name', 'description', 'input_schema')}
                for t in tools
            ]

        resp = self.anthropic_client.messages.create(**kwargs)

        raw_text = ''
        all_tools: list[dict] = []
        for block in resp.content:
            if block.type == 'text':
                raw_text += block.text
            elif block.type == 'tool_use':
                all_tools.append({'name': block.name, 'args': block.input})

        result = {
            'raw_response': raw_text,
            'input_tokens': resp.usage.input_tokens,
            'output_tokens': resp.usage.output_tokens,
            'tools_called': all_tools,
        }
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']
        return result

    def _call_google(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        from google.genai import types

        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)

        gemini_contents = []
        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg['content'])],
                )
            )

        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1024,
            response_mime_type='application/json' if not tools else None,
        )
        if temperature > 0:
            gen_config.temperature = temperature

        # Gemini API requires explicit thinking config
        if 'gemini-3-pro' in model_id or 'gemini-3.1-pro' in model_id:
            # Gemini Pro requires thinking; use minimal budget
            gen_config.thinking_config = types.ThinkingConfig(thinking_budget=128)
        elif 'gemini-3' in model_id:
            # Gemini Flash: disable thinking
            gen_config.thinking_config = types.ThinkingConfig(thinking_budget=0)

        if tools:
            gemini_tools = []
            for t in tools:
                fn = types.FunctionDeclaration(
                    name=t['name'],
                    description=t.get('description', ''),
                    parameters=t.get('input_schema'),
                )
                gemini_tools.append(fn)
            gen_config.tools = [types.Tool(function_declarations=gemini_tools)]

        resp = self.google_client.models.generate_content(
            model=model_id,
            contents=gemini_contents,
            config=gen_config,
        )

        raw_text = resp.text or ''
        result: dict[str, Any] = {
            'raw_response': raw_text,
            'input_tokens': getattr(resp.usage_metadata, 'prompt_token_count', 0) or 0,
            'output_tokens': getattr(resp.usage_metadata, 'candidates_token_count', 0) or 0,
        }

        # Check for tool calls
        all_tools: list[dict] = []
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    all_tools.append({
                        'name': part.function_call.name,
                        'args': dict(part.function_call.args) if part.function_call.args else {},
                    })
        result['tools_called'] = all_tools
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']

        return result

    def _call_gemma(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Call Gemma via Google genai API.

        Gemma on the Google API does NOT support:
          - system_instruction  → baked into the first user message
          - response_mime_type  → prompt asks for JSON; _parse_flows strips fences
          - native tool use     → tools inlined into prompt; response parsed as JSON
          - thinking_config     → no thinking modes available
          - output token counts → candidates_token_count returns None

        Use provider='gemma' (NOT 'google') — this is an API routing key,
        not a taxonomy label.  Conceptually Google's low-tier model, but
        uses a separate API path.
        """
        from google.genai import types

        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)

        # If tools are provided, inline their descriptions into the system prompt
        # so the model can "call" them via JSON output.
        effective_system = system_prompt
        if tools:
            tool_lines = []
            for t in tools:
                params = t.get('input_schema', {}).get('properties', {})
                required = t.get('input_schema', {}).get('required', [])
                param_strs = []
                for pname, pspec in params.items():
                    req_tag = ', required' if pname in required else ''
                    param_strs.append(
                        f'  - {pname} ({pspec.get("type", "string")}{req_tag}): '
                        f'{pspec.get("description", "")}'
                    )
                tool_lines.append(
                    f'Tool: {t["name"]}\n'
                    f'Description: {t.get("description", "")}\n'
                    f'Parameters:\n' + '\n'.join(param_strs)
                )
            effective_system += (
                '\n\nYou have the following tools available:\n\n'
                + '\n\n'.join(tool_lines)
                + '\n\nTo use a tool, respond ONLY with valid JSON:\n'
                '{"tool": "<tool_name>", "args": {<arg_name>: <value>, ...}}'
            )

        # Build contents with system prompt baked into the first user message
        gemma_contents = []
        system_prepended = False
        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            content = msg['content']
            if role == 'user' and not system_prepended:
                content = f'[INSTRUCTIONS]\n{effective_system}\n\n[USER MESSAGE]\n{content}'
                system_prepended = True
            gemma_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        gen_config = types.GenerateContentConfig(max_output_tokens=1024)
        if temperature > 0:
            gen_config.temperature = temperature

        resp = self.google_client.models.generate_content(
            model=model_id,
            contents=gemma_contents,
            config=gen_config,
        )

        raw_text = resp.text or ''
        # Gemma often wraps JSON in markdown fences — strip them
        cleaned = raw_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        result: dict[str, Any] = {
            'raw_response': cleaned,
            'input_tokens': getattr(resp.usage_metadata, 'prompt_token_count', 0) or 0,
            'output_tokens': getattr(resp.usage_metadata, 'candidates_token_count', 0) or 0,
        }

        # Parse simulated tool calls from JSON response
        result['tools_called'] = []
        if tools:
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'tool' in parsed:
                    tool_entry = {'name': parsed['tool'], 'args': parsed.get('args', {})}
                    result['tools_called'] = [tool_entry]
                    result['tool_called'] = tool_entry['name']
                    result['tool_args'] = tool_entry['args']
            except (json.JSONDecodeError, KeyError):
                pass  # No valid tool call found; caller handles missing keys

        return result

    def _call_openai(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)

        oai_messages = [{'role': 'system', 'content': system_prompt}]
        for msg in messages:
            oai_messages.append({'role': msg['role'], 'content': msg['content']})

        kwargs: dict[str, Any] = {
            'model': model_id,
            'messages': oai_messages,
            'max_completion_tokens': config.get('max_tokens', 1024),
        }

        if temperature > 0:
            kwargs['temperature'] = temperature

        if tools:
            kwargs['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t.get('description', ''),
                        'parameters': t.get('input_schema', {}),
                    },
                }
                for t in tools
            ]

        resp = self.openai_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        raw_text = choice.message.content or ''
        result: dict[str, Any] = {
            'raw_response': raw_text,
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
        }

        all_tools: list[dict] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                all_tools.append({
                    'name': tc.function.name,
                    'args': json.loads(tc.function.arguments) if tc.function.arguments else {},
                })
        result['tools_called'] = all_tools
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']

        return result

    def _call_openrouter(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Call a model via OpenRouter's OpenAI-compatible API."""
        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)

        or_messages = [{'role': 'system', 'content': system_prompt}]
        for msg in messages:
            or_messages.append({'role': msg['role'], 'content': msg['content']})

        kwargs: dict[str, Any] = {
            'model': model_id,
            'messages': or_messages,
            'max_completion_tokens': config.get('max_tokens', 4096),
        }

        if temperature > 0:
            kwargs['temperature'] = temperature

        if tools:
            kwargs['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t.get('description', ''),
                        'parameters': t.get('input_schema', {}),
                    },
                }
                for t in tools
            ]

        resp = self.openrouter_client.chat.completions.create(**kwargs)
        if not resp or not resp.choices:
            raise RuntimeError('openrouter server returned empty choices (transient error)')
        choice = resp.choices[0]

        raw_text = choice.message.content or ''
        result: dict[str, Any] = {
            'raw_response': raw_text,
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
        }

        all_tools: list[dict] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                all_tools.append({
                    'name': tc.function.name,
                    'args': json.loads(tc.function.arguments) if tc.function.arguments else {},
                })
        result['tools_called'] = all_tools
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']

        return result

    def _call_qwen(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Call Qwen via Alibaba Cloud's OpenAI-compatible endpoint."""
        import openai

        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)
        api_key = os.environ.get('TOGETHER_API_KEY') or os.environ.get('QWEN_API_KEY')
        base_url = os.environ.get('QWEN_BASE_URL', 'https://api.together.xyz/v1')

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        qwen_messages = [{'role': 'system', 'content': system_prompt}]
        for msg in messages:
            qwen_messages.append({'role': msg['role'], 'content': msg['content']})

        is_thinking = 'thinking' in model_id.lower()
        default_max = 8192 if is_thinking else 1024
        kwargs: dict[str, Any] = {
            'model': model_id,
            'messages': qwen_messages,
            'max_completion_tokens': config.get('max_tokens', default_max),
        }

        # Thinking models don't support temperature
        if temperature > 0 and not is_thinking:
            kwargs['temperature'] = temperature

        if tools:
            kwargs['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t.get('description', ''),
                        'parameters': t.get('input_schema', {}),
                    },
                }
                for t in tools
            ]

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        raw_text = choice.message.content or ''

        # Qwen thinking models may wrap output in <think>...</think> tags
        if is_thinking and '<think>' in raw_text:
            # Extract content after </think> tag
            parts = raw_text.split('</think>')
            if len(parts) > 1:
                raw_text = parts[-1].strip()
            else:
                # No closing tag — try to extract JSON from the full text
                match = re.search(r'\{[^{}]*"intent"[^{}]*\}', raw_text)
                if match:
                    raw_text = match.group()

        result: dict[str, Any] = {
            'raw_response': raw_text,
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
        }

        all_tools: list[dict] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                all_tools.append({
                    'name': tc.function.name,
                    'args': json.loads(tc.function.arguments) if tc.function.arguments else {},
                })
        result['tools_called'] = all_tools
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']

        return result

    def _call_deepseek(
        self, config: dict, system_prompt: str, messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        import httpx

        model_id = config['model_id']
        temperature = config.get('temperature', 0.0)
        api_key = os.environ['DEEPSEEK_API_KEY']

        ds_messages = [{'role': 'system', 'content': system_prompt}]
        for msg in messages:
            ds_messages.append({'role': msg['role'], 'content': msg['content']})

        is_reasoner = 'reasoner' in model_id.lower()
        default_max = 8192 if is_reasoner else 1024
        body: dict[str, Any] = {
            'model': model_id,
            'messages': ds_messages,
            'max_tokens': config.get('max_tokens', default_max),
        }
        # Reasoner does not support temperature
        if temperature > 0 and not is_reasoner:
            body['temperature'] = temperature

        if tools:
            body['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': t['name'],
                        'description': t.get('description', ''),
                        'parameters': t.get('input_schema', {}),
                    },
                }
                for t in tools
            ]

        resp = httpx.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json=body,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data['choices'][0]
        raw_text = choice['message'].get('content', '') or ''

        # DeepSeek Reasoner: fall back to reasoning_content if content is empty
        if not raw_text.strip() and is_reasoner:
            reasoning = choice['message'].get('reasoning_content', '') or ''
            if reasoning:
                # Try to extract JSON from the reasoning chain
                match = re.search(r'\{[^{}]*"intent"[^{}]*\}', reasoning)
                if match:
                    raw_text = match.group()

        usage = data.get('usage', {})
        result: dict[str, Any] = {
            'raw_response': raw_text,
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
        }

        all_tools: list[dict] = []
        tool_calls = choice['message'].get('tool_calls')
        if tool_calls:
            for tc in tool_calls:
                all_tools.append({
                    'name': tc['function']['name'],
                    'args': json.loads(tc['function']['arguments']) if tc['function'].get('arguments') else {},
                })
        result['tools_called'] = all_tools
        if all_tools:
            result['tool_called'] = all_tools[0]['name']
            result['tool_args'] = all_tools[0]['args']

        return result

    # ── Response parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_flows(raw_response: str) -> dict:
        """Extract flows list and reasoning from JSON response.

        Handles the new format: {"reasoning": "...", "flows": ["flow_name"]}
        Also handles legacy format: {"flow": "flow_name"}
        """
        def _normalize(obj: dict) -> dict:
            # New format: {"reasoning": ..., "flows": [...]}
            flows = obj.get('flows')
            reasoning = obj.get('reasoning', '')

            if flows is not None:
                # Normalize string to list
                if isinstance(flows, str):
                    flows = [flows]
                # Lowercase all flow names
                flows = [f.lower().strip() for f in flows if f]
                return {'flows': flows, 'reasoning': reasoning}

            # Legacy format: {"flow": "flow_name"}
            flow = obj.get('flow') or obj.get('flow_name')
            if flow:
                return {'flows': [flow.lower().strip()], 'reasoning': reasoning}

            return {'flows': [], 'reasoning': reasoning}

        try:
            cleaned = raw_response.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            return _normalize(json.loads(cleaned))
        except (json.JSONDecodeError, AttributeError):
            # Try to find JSON object in the response
            match = re.search(r'\{[^{}]*\}', raw_response)
            if match:
                try:
                    return _normalize(json.loads(match.group()))
                except (json.JSONDecodeError, AttributeError):
                    pass
            log.warning('Failed to parse flows from response: %s', raw_response[:200])
            return {'flows': [], 'reasoning': ''}
