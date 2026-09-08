# Team onboarding

## One-time owner setup

1. Share the public starter-kit URL. Team members do not need GitHub collaborator or write access.
2. Each teammate may use their own Google Drive; a shared folder is optional.
3. Put the public competition images in any one folder in that Drive. An `images/` subfolder is optional.
4. On the first run, leave `DRIVE_ROOT` blank in the Colab notebook. The agent searches the logged-in user's Drive for one folder containing roughly 3,000 images, then writes `DATASET_MANIFEST.yaml` beside that folder. It asks for a choice if several folders match.
5. To initialize a known folder manually instead:

```bash
python scripts/init_drive_dataset.py /content/drive/MyDrive/any-uploaded-folder
```

If a labeled dev set exists, place it anywhere under the same root and add `--labels-file labels/dev_labels.csv` when initializing.

## Per-team-member setup

1. Clone the public repository. This is a local working copy, not a shared workspace.
2. Give the preferred coding agent the repository folder. Codex reads `AGENTS.md`; Claude Code reads `CLAUDE.md`; Antigravity/Gemini reads `GEMINI.md`.
3. No GitHub secret is needed for the public starter kit. An `ITDA_GITHUB_TOKEN` secret is optional only when a team member wants to clone a private fork instead.
4. Ask the agent to start an experiment. It should run:

```bash
python scripts/start_experiment.py --owner <github-handle> --name <experiment-slug>
```

This creates a local config under `configs/experiments/`. Add `--branch` only if a local branch helps the individual's own work; never push it.

## Slack policy

Slack never posts by default. Every run writes `team_report.md` and `slack_message.md`; a real message is sent only when `--send` is passed to `notify_slack.py`, or `SEND_SLACK=True` is explicitly selected in the Colab notebook.
