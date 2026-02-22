from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LeagueRating:
  policy_id: str
  elo: float = 1000.0
  games: int = 0
  wins: int = 0
  losses: int = 0
  draws: int = 0


def expected_score(elo_a: float, elo_b: float) -> float:
  return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float, score_a: float, k_factor: float = 24.0) -> tuple[float, float]:
  exp_a = expected_score(elo_a, elo_b)
  exp_b = 1.0 - exp_a
  new_a = elo_a + k_factor * (score_a - exp_a)
  new_b = elo_b + k_factor * ((1.0 - score_a) - exp_b)
  return new_a, new_b


def apply_match_result(
  rating_a: LeagueRating,
  rating_b: LeagueRating,
  *,
  score_a: float,
  k_factor: float = 24.0,
) -> tuple[LeagueRating, LeagueRating]:
  new_a, new_b = update_elo(rating_a.elo, rating_b.elo, score_a, k_factor=k_factor)
  rating_a.elo = new_a
  rating_b.elo = new_b
  rating_a.games += 1
  rating_b.games += 1
  if score_a >= 0.999:
    rating_a.wins += 1
    rating_b.losses += 1
  elif score_a <= 0.001:
    rating_a.losses += 1
    rating_b.wins += 1
  else:
    rating_a.draws += 1
    rating_b.draws += 1
  return rating_a, rating_b


def rank_table(ratings: list[LeagueRating]) -> list[LeagueRating]:
  return sorted(ratings, key=lambda r: (-r.elo, -r.wins, r.losses, r.policy_id))


def save_ratings(path: Path, ratings: list[LeagueRating]) -> None:
  payload = {"ratings": [asdict(item) for item in ratings]}
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_ratings(path: Path) -> list[LeagueRating]:
  if not path.exists():
    return []
  raw = json.loads(path.read_text(encoding="utf-8"))
  items = raw.get("ratings", [])
  out = []
  for item in items:
    out.append(
      LeagueRating(
        policy_id=str(item["policy_id"]),
        elo=float(item.get("elo", 1000.0)),
        games=int(item.get("games", 0)),
        wins=int(item.get("wins", 0)),
        losses=int(item.get("losses", 0)),
        draws=int(item.get("draws", 0)),
      )
    )
  return out
