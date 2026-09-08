# Team onboarding

## One-time owner setup

1. Invite every teammate to the private GitHub repository as a collaborator.
2. Create one shared Drive folder, for example `ITDA_OCR`.
3. Put the public competition images under `ITDA_OCR/images/`.
4. In Colab, mount Drive and run the dataset initializer once:

```bash
python scripts/init_drive_dataset.py /content/drive/MyDrive/ITDA_OCR
```

If a labeled dev set exists, place it under `labels/` and add `--labels-file labels/dev_labels.csv` when initializing.

## Per-team-member setup

1. Clone the repository after collaborator access is granted.
2. Give the preferred coding agent the repository folder. Codex reads `AGENTS.md`; Claude Code reads `CLAUDE.md`; Antigravity/Gemini reads `GEMINI.md`.
3. For Colab, create an `ITDA_GITHUB_TOKEN` secret with read access to this private repository. Google login alone does not grant private GitHub access.
4. Ask the agent to start an experiment. It should run:

```bash
python scripts/start_experiment.py --owner <github-handle> --name <experiment-slug>
```

This creates `feature/<github-handle>-<experiment-slug>` and a config under `configs/experiments/`.

## Slack policy

Slack never posts by default. Every run writes `slack_message.md`; a real message is sent only when `--send` is passed to `notify_slack.py`, or `SEND_SLACK=True` is explicitly selected in the Colab notebook.
