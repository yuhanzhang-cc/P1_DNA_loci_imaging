# AGENTS.md

## Project workflow
This repository is developed by a small team with direct push access.
Do not assume PR-based workflow.
Keep changes minimal, reviewable, and easy to merge.

## Branch rules
- Do not work directly on `main` unless explicitly asked.
- Prefer task branches such as `feat/...` or `fix/...`.
- When acting as Codex, prefer a dedicated temporary branch such as `codex/...` if the user asks for non-trivial edits.
- Before making large edits, first inspect the current branch and recent changes.

## Change policy
- Make the smallest change that fully solves the task.
- Do not refactor unrelated files.
- Do not rename files, move directories, or rewrite module structure unless the task explicitly requires it.
- Preserve the existing coding style unless there is a clear local convention to follow.

## Environment policy
- This project uses a shared local development environment across the human developer and Codex CLI.
- Prefer repository-tracked environment definition files, such as `environment.yml` or `requirements.txt`.
- Do not commit local environment directories such as `.venv/`, `venv/`, `env/`, or machine-specific cache folders.
- If a dependency change is necessary, update the environment definition file and explain why.

## Commands
- Before finalizing substantial code changes, run the smallest relevant validation command available.
- Prefer targeted tests over full expensive runs unless the user explicitly asks for a full run.
- If no tests exist, at least run lightweight syntax or import checks when appropriate.

## Git safety
- Never discard user changes.
- Never run destructive Git commands such as hard reset, force checkout, or history rewrites unless the user explicitly asks.
- If the working tree is dirty, first summarize the state before making risky changes.
- If conflicts appear, stop and clearly describe the conflict files and the safe next step.

## Data and secrets
- Never commit secrets, tokens, passwords, private keys, `.env` contents, or raw confidential datasets.
- Avoid modifying large generated files unless the task explicitly targets them.

## Communication style
- First state what files you plan to change.
- After editing, summarize:
  1. what changed,
  2. why it changed,
  3. what should be checked in VS Code diff before commit.

## For this repository specifically
- Prefer edits that are easy to inspect in VS Code Source Control.
- Keep commits focused and small.
- When changing dependencies, configs, or scripts used by collaborators, call that out explicitly.