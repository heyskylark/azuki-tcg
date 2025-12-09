from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import contextlib
import io
import json
import time

import torch
import pufferlib.vector
import pufferlib.pytorch

from training_utils import (
  DEFAULT_CONFIG_PATH,
  build_policy,
  build_vecenv,
  install_tcg_sampler,
  load_training_config,
)


def _unwrap_base_env(env: Any):
  current = getattr(env, "env", env)
  seen = set()
  while hasattr(current, "env"):
    nxt = getattr(current, "env")
    if nxt is current or nxt in seen:
      break
    seen.add(current)
    current = nxt
  return current


def _write_frame(handle, frame: str, episode: int, step: int, clear_prefix: str = ""):
  if clear_prefix:
    handle.write(clear_prefix)
  handle.write(f"\n[episode {episode} step {step}]\n")
  handle.write(frame)
  if not frame.endswith("\n"):
    handle.write("\n")
  handle.flush()


def run_playback(
  *,
  checkpoint: Path,
  config_path: Path = DEFAULT_CONFIG_PATH,
  episodes: int = 1,
  max_steps: int | None = 200,
  device: str = "cpu",
  output_path: Path | None = None,
  render_mode: str = "ansi",
  seed: int | None = None,
  asciicast: bool = False,
  cast_output_path: Path | None = None,
  cast_interval: float = 0.05,
  cast_width: int = 120,
  cast_height: int = 40,
  clear_frames: bool = True,
):
  checkpoint = Path(checkpoint)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  trainer_args = load_training_config(config_path, [])
  trainer_args["train"]["device"] = device
  install_tcg_sampler()

  vecenv = build_vecenv(
    trainer_args,
    backend=pufferlib.vector.Serial,
    num_envs=1,
    seed=seed if seed is not None else 0,
  )
  policy = build_policy(vecenv, trainer_args)
  use_rnn = trainer_args["train"].get("use_rnn", True)

  # Warm up once so scalar normalizer buffers exist before loading.
  vecenv.async_reset(seed=seed if seed is not None else 0)
  warm_obs, _, _, _, _, _, warm_masks = vecenv.recv()
  warm_state = {"mask": torch.as_tensor(warm_masks, device=device)}
  if use_rnn:
    warm_state["lstm_h"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
    warm_state["lstm_c"] = torch.zeros(vecenv.num_agents, policy.hidden_size, device=device)
  with torch.no_grad():
    policy.forward_eval(torch.as_tensor(warm_obs, device=device), warm_state)

  state_dict = torch.load(checkpoint, map_location=device)
  try:
    policy.load_state_dict(state_dict)
  except RuntimeError:
    policy.load_state_dict(state_dict, strict=False)
  policy.eval()

  base_env = _unwrap_base_env(vecenv.envs[0])
  frames_written = 0
  output_handle = open(output_path, "w") if output_path else sys.stdout
  cast_frames = []
  cast_path = None
  if asciicast:
    if cast_output_path is not None:
      cast_path = Path(cast_output_path)
    elif output_path is not None:
      cast_path = Path(output_path).with_suffix(".cast")
    else:
      cast_path = Path("playback.cast")

  def render_frame(target_env):
    # BFS through wrapper stack to find anything that can give us a string frame.
    queue = [target_env]
    seen = set()
    while queue:
      env_obj = queue.pop(0)
      if id(env_obj) in seen:
        continue
      seen.add(id(env_obj))
      if hasattr(env_obj, "env"):
        queue.append(getattr(env_obj, "env"))
      if hasattr(env_obj, "aec_env"):
        queue.append(getattr(env_obj, "aec_env"))

      render_fn = getattr(env_obj, "render", None)
      if not callable(render_fn):
        continue

      # Try with mode first, then without; capture stdout so we don't lose frames that print.
      for call_with_mode in (True, False):
        try:
          with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            if call_with_mode:
              frame = render_fn(mode=render_mode)
            else:
              frame = render_fn()
            printed = buf.getvalue()
        except TypeError:
          # Signature might not accept mode; fall through to no-mode.
          continue
        except Exception:
          continue

        if frame is not None:
          return frame
        if printed.strip():
          return printed
    return None

  def record_frame(frame_text: str, episode: int, step: int):
    nonlocal frames_written
    if frame_text is None:
      return
    clear_prefix = "\x1b[2J\x1b[H" if clear_frames else ""
    payload_text = frame_text
    if output_handle:
      _write_frame(output_handle, payload_text, episode, step, clear_prefix=clear_prefix)
    if asciicast:
      payload_body = payload_text if payload_text.endswith("\n") else payload_text + "\n"
      payload = clear_prefix + payload_body
      timestamp = round(len(cast_frames) * cast_interval, 4)
      cast_frames.append([timestamp, "o", payload])
    frames_written += 1

  try:
    for episode_idx in range(episodes):
      if use_rnn:
        state = {
          "lstm_h": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
          "lstm_c": torch.zeros(vecenv.num_agents, policy.hidden_size, device=device),
        }
      else:
        state = {}

      vecenv.async_reset(seed=(seed or 0) + episode_idx)
      obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()

      step_idx = 0
      frame = render_frame(base_env)
      record_frame(frame, episode_idx, step_idx)

      done = False
      while not done and (max_steps is None or step_idx < max_steps):
        obs_tensor = torch.as_tensor(obs, device=device)
        step_state = {"mask": torch.as_tensor(masks, device=device)}
        if use_rnn:
          step_state["lstm_h"] = state["lstm_h"]
          step_state["lstm_c"] = state["lstm_c"]

        with torch.no_grad():
          logits, _ = policy.forward_eval(obs_tensor, step_state)
          action, _, _ = pufferlib.pytorch.sample_logits(logits)

        if use_rnn:
          state["lstm_h"] = step_state["lstm_h"]
          state["lstm_c"] = step_state["lstm_c"]

        vecenv.send(action.cpu().numpy())
        obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
        step_idx += 1

        frame = render_frame(base_env)
        record_frame(frame, episode_idx, step_idx)

        done = vecenv.envs[0].done
  finally:
    if output_path:
      output_handle.close()
    vecenv.close()

  cast_written = None
  if asciicast and cast_path is not None:
    header = {
      "version": 2,
      "width": cast_width,
      "height": cast_height,
      "timestamp": int(time.time()),
      "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    }
    lines = [json.dumps(header)]
    frames_to_write = cast_frames if cast_frames else [[0, "o", ""]]
    for entry in frames_to_write:
      lines.append(json.dumps(entry))
    cast_path.write_text("\n".join(lines) + "\n")
    cast_written = cast_path

  return {"frames": frames_written, "output": output_path, "cast": cast_written}


def parse_args():
  parser = argparse.ArgumentParser(
    description="Render a saved Azuki TCG checkpoint as ANSI frames you can pipe to a recorder."
  )
  parser.add_argument(
    "--checkpoint",
    required=True,
    type=Path,
    help="Path to the saved model (.pt) produced by train.py.",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=DEFAULT_CONFIG_PATH,
    help="Training config to mirror env setup (defaults to the main azuki.ini).",
  )
  parser.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="How many episodes to roll out.",
  )
  parser.add_argument(
    "--max-steps",
    type=int,
    default=200,
    help="Maximum steps to render per episode (guardrail for long games).",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to load and run the policy on (cpu or cuda).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    help="Optional file to write ANSI frames into. Defaults to stdout for piping to ttyrec/asciinema.",
  )
  parser.add_argument(
    "--asciicast",
    action="store_true",
    help="Also emit an asciicast v2 JSON (.cast).",
  )
  parser.add_argument(
    "--cast-output",
    type=Path,
    help="Optional path for the .cast file (defaults to output with .cast suffix or playback.cast).",
  )
  parser.add_argument(
    "--cast-interval",
    type=float,
    default=0.05,
    help="Seconds between frames in the asciicast timeline.",
  )
  parser.add_argument(
    "--cast-width",
    type=int,
    default=120,
    help="Terminal width to record in the asciicast header.",
  )
  parser.add_argument(
    "--cast-height",
    type=int,
    default=40,
    help="Terminal height to record in the asciicast header.",
  )
  parser.add_argument(
    "--no-clear-frames",
    action="store_true",
    help="Disable sending a clear-screen before each frame (useful if you prefer raw appended text).",
  )
  parser.add_argument(
    "--cast-only",
    action="store_true",
    help="Skip writing ANSI output and only emit the .cast file.",
  )
  parser.add_argument(
    "--render-mode",
    type=str,
    default="ansi",
    choices=["ansi", "human"],
    help="Render mode to request from the env.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    help="Episode seed. If omitted, uses 0 and increments per episode.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  run_playback(
    checkpoint=args.checkpoint,
    config_path=args.config,
    episodes=args.episodes,
    max_steps=args.max_steps,
    device=args.device,
    output_path=None if args.cast_only else args.output,
    render_mode=args.render_mode,
    seed=args.seed,
    asciicast=args.asciicast,
    cast_output_path=args.cast_output,
    cast_interval=args.cast_interval,
    cast_width=args.cast_width,
    cast_height=args.cast_height,
    clear_frames=not args.no_clear_frames,
  )


if __name__ == "__main__":
  main()
