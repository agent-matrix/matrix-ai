# Matrix AI v2: deliberation, not authority

The v2 API adds `POST /v2/deliberate`.

The service:

1. receives a goal, scoped context and available capabilities,
2. asks MatrixLLM for materially different candidate strategies,
3. requires each executable step to declare success criteria and verifiers,
4. returns all candidates plus one selected proposal and uncertainty.

It does **not** authorize, fund or execute the plan. Those decisions belong to
Guardian, Treasury and Matrix OS.

## Model routing

V2 deliberation uses MatrixLLM's OpenAI-compatible gateway through:

- `MATRIX_LLM_BASE_URL`
- `MATRIX_LLM_API_KEY`
- `MATRIX_LLM_MODEL`

This removes provider-specific routing from the new reasoning path.
