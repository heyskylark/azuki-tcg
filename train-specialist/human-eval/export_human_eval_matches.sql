-- One row per scheduled human-evaluation match with its frozen assignment
-- (deck source / premade deck / gate / leader / seeds), outcome and ratings.
-- Usage: psql --csv -f train-specialist/human-eval/export_human_eval_matches.sql > results.csv
SELECT
  s.id AS session_id,
  s.schedule_seed,
  s.games_per_model,
  s.revealed_at,
  m.ordinal,
  m.id AS match_id,
  m.model_key_snapshot AS model_key,
  m.checkpoint_sha256_snapshot AS checkpoint_sha256,
  m.deck_source,
  m.premade_deck_slug,
  m.gate_card_code,
  m.leader_card_code,
  m.ai_slot,
  m.starting_player,
  m.draft_seed,
  m.battle_seed,
  m.status,
  a.deck_hash,
  CASE
    WHEN r.id IS NULL THEN NULL
    WHEN r.winner_id IS NULL THEN 'DRAW'
    WHEN r.winner_id = s.reviewer_id THEN 'HUMAN_WIN'
    ELSE 'AI_WIN'
  END AS outcome,
  r.win_type,
  r.total_turns,
  r.duration_seconds,
  n.opponent_strength_rating,
  n.decision_quality_rating,
  n.deck_coherence_rating,
  n.human_likeness_rating,
  n.match_enjoyment_rating,
  n.model_guess,
  n.guess_confidence,
  n.notes,
  (SELECT count(*) FROM human_evaluation_actions x WHERE x.match_id = m.id) AS recorded_actions
FROM human_evaluation_matches m
JOIN human_evaluation_sessions s ON s.id = m.session_id
LEFT JOIN human_evaluation_deck_artifacts a ON a.match_id = m.id
LEFT JOIN match_results r ON r.id = m.match_result_id
LEFT JOIN human_evaluation_annotations n ON n.match_id = m.id
ORDER BY s.created_at, m.ordinal;
