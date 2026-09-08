# Team onboarding

## One-time owner setup

1. Publish the starter kit as a shareable GitHub page/release, or grant every teammate clone access while it remains private. Team members do not need write access.
2. Each teammate may use their own Google Drive; a shared folder is optional.
3. Put the public competition images in any one folder in that Drive. An `images/` subfolder is optional.
4. On the first run, leave `DRIVE_ROOT` blank in the Colab notebook. The agent searches the logged-in user's Drive for one folder containing roughly 3,000 images, then writes `DATASET_MANIFEST.yaml` beside that folder. It asks for a choice if several folders match.
5. To initialize a known folder manually instead:

```bash
python scripts/init_drive_dataset.py /content/drive/MyDrive/any-uploaded-folder
```

If a labeled dev set exists, place it anywhere under the same root and add `--labels-file labels/dev_labels.csv` when initializing.

## Per-team-member setup

1. Clone the repository after clone access is granted. This is a local working copy, not a shared workspace.
2. Give the preferred coding agent the repository folder. Codex reads `AGENTS.md`; Claude Code reads `CLAUDE.md`; Antigravity/Gemini reads `GEMINI.md`.
3. For Colab, create an `ITDA_GITHUB_TOKEN` secret with read access to this private repository. Google login alone does not grant private GitHub access.
4. Ask the agent to start an experiment. It should run:

```bash
python scripts/start_experiment.py --owner <github-handle> --name <experiment-slug>
```

This creates a local config under `configs/experiments/`. Add `--branch` only if a local branch helps the individual's own work; never push it.

## Slack policy

Slack never posts by default. Every run writes `team_report.md` and `slack_message.md`; a real message is sent only when `--send` is passed to `notify_slack.py`, or `SEND_SLACK=True` is explicitly selected in the Colab notebook.
