"""Compatibility entry point for google-colab-cli 0.6 with jupyter-kernel-client & auto-drivemount."""
import builtins
import datetime
import io
import json
import os
import subprocess
import sys
import time

import jupyter_kernel_client

if not hasattr(jupyter_kernel_client, "KernelClient") and hasattr(jupyter_kernel_client, "JupyterKernelClient"):
    jupyter_kernel_client.KernelClient = jupyter_kernel_client.JupyterKernelClient

import colab_cli.commands.automation as auto_mod
from colab_cli.utils import get_status_code, render_display_data
import typer

def auto_run_automation(
    name: str,
    op: str,
    code: str,
    allow_stdin: bool = False,
    path: str = None,
    timeout: float = None,
):
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime
    from colab_cli.auth import get_credentials

    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    runtime = ColabRuntime(s.url, s.token, session_name=s.name, history=state.history)

    def drivefs_hook(deserialize_msg, wsclient):
        content = deserialize_msg.get("content", {})
        if content.get("request", {}).get("authType") == "dfs_ephemeral":
            msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
            url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"
            params = {
                "authuser": "0",
                "authtype": "dfs_ephemeral",
                "version": "2",
                "dryrun": "true",
                "propagate": "true",
                "record": "false",
            }
            typer.echo(f"\n[colab] Intercepted Drive Auth Request. Connecting to {state.client.colab_domain}...")
            creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
            resp = creds.request("GET", url, params=params)
            token = (
                json.loads(resp.text.split("\n", 1)[-1]).get("token")
                if get_status_code(resp) == 200
                else None
            )
            headers = {"x-goog-colab-token": token}
            resp = creds.request(
                "POST",
                url,
                params=params,
                headers=headers,
                files={"file_id": (None, "empty.ipynb")},
            )
            data = json.loads(resp.text.split("\n", 1)[-1])

            if not data.get("success"):
                uri = data.get("unauthorized_redirect_uri")
                typer.echo(f"\n[colab] REQUIRED: Google Drive Authorization needed.\nOpening browser:\n{uri}\n")
                try:
                    subprocess.run(["open", uri], check=False)
                except Exception:
                    pass

                typer.echo("[colab] Waiting for Google Drive consent approval (polling)...")
                for _ in range(60):
                    time.sleep(2)
                    params["dryrun"] = "false"
                    resp = creds.request(
                        "POST",
                        url,
                        params=params,
                        headers=headers,
                        files={"file_id": (None, "empty.ipynb")},
                    )
                    if get_status_code(resp) == 200:
                        break
            else:
                params["dryrun"] = "false"
                resp = creds.request(
                    "POST",
                    url,
                    params=params,
                    headers=headers,
                    files={"file_id": (None, "empty.ipynb")},
                )

            if get_status_code(resp) == 200:
                typer.echo("[colab] Credentials propagated. Resuming mount...")
                state.history.log_event(s.name, "drive_auth_success", {})
                reply = wsclient.session.msg(
                    "input_reply",
                    {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
                )
                if "header" in deserialize_msg:
                    reply["parent_header"] = deserialize_msg["header"]
                wsclient.stdin_channel.send(reply)
                return True
            else:
                typer.echo(f"[colab] Error propagating credentials: {get_status_code(resp)} {resp.text}")
                return False
        return False

    runtime.colab_request_hook = drivefs_hook
    try:
        s.running = f"automation({op})"
        s.last_execution = (
            f"automation:{op}",
            None,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        state.store.add(s)

        outputs = runtime.execute_code(code, allow_stdin=allow_stdin, timeout=timeout or 600)
        for out in outputs:
            if "text" in out:
                sys.stdout.write(out["text"])
            elif out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                sys.stderr.write(f"{ename}: {evalue}\n")
    finally:
        s.running = None
        state.store.add(s)
        runtime.stop()

auto_mod.run_automation = auto_run_automation

from colab_cli.cli import main

if __name__ == "__main__":
    sys.exit(main())
