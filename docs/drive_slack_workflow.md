# Drive and Slack workflow

The shared Drive dataset is discovered through `DATASET_MANIFEST.yaml`, not a fuzzy folder-name search. This prevents two team members from accidentally running different copies of the dataset.

After a run, the pipeline writes `architecture.json` and `architecture.md` alongside `metrics.json`. `scripts/notify_slack.py` includes the architecture summary in a Slack-ready message.

```bash
python scripts/notify_slack.py runs/my_experiment \
  --artifact-url "https://drive.google.com/..." \
  --github-url "https://github.com/Great-Grace/itda-ocr-lab/commit/..."
```

The command creates `slack_message.md` and prints the message. Add `--send` only after `SLACK_WEBHOOK_URL` has been configured in the local/Colab secret store. Raw images and OCR logs are not posted to Slack; the message links to the Drive artifact folder.
