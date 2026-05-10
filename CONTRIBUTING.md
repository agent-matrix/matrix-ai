# Contributing

Thanks for your interest in contributing to this Agent-Matrix repository.

## Quick start

1. Fork the repository and create a topic branch from the latest default branch.
2. Make focused, additive changes with tests where applicable.
3. Run the standard local checks documented in `README.md` (lint, test, build).
4. Open a pull request using the pull-request template.

## Branch naming

Use one of:

- `feat/<short-description>`
- `fix/<short-description>`
- `chore/<short-description>`
- `docs/<short-description>`
- `refactor/<short-description>`

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` user-facing feature
- `fix: ...` user-facing bug fix
- `chore: ...` non-functional change
- `docs: ...` documentation change
- `refactor: ...` non-functional refactor
- `test: ...` test-only change

## Pull request expectations

- Keep PRs small and focused.
- Reference any related issue: `Closes #123`.
- Update `CHANGELOG.md` under `Unreleased` when behaviour or APIs change.
- Ensure CI is green before requesting review.
- Do not commit secrets, credentials, or private endpoints.

## Alive-system contracts

This repository participates in the Agent-Matrix alive system. When changes
touch any of the following surfaces, please document them in the PR:

- Health endpoints or commands
- Emitted or consumed events
- Governance metadata (risk, permissions, approvals)
- Economics metadata (cost, budget)

See `docs/alive-integration.md` (if present) for the integration contract.

## Code of conduct

Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please do not file public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the responsible disclosure process.
