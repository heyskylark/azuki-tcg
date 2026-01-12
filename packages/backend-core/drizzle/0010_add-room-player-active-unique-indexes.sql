-- Partial unique indexes to enforce one active room per player
-- Active statuses: WAITING_FOR_PLAYERS, DECK_SELECTION, READY_CHECK, STARTING, IN_MATCH
-- Inactive statuses (excluded): COMPLETED, ABORTED, CLOSED

CREATE UNIQUE INDEX rooms_player0_active_idx ON rooms (player0_id)
WHERE status NOT IN ('COMPLETED', 'ABORTED', 'CLOSED') AND player0_id IS NOT NULL;--> statement-breakpoint

CREATE UNIQUE INDEX rooms_player1_active_idx ON rooms (player1_id)
WHERE status NOT IN ('COMPLETED', 'ABORTED', 'CLOSED') AND player1_id IS NOT NULL;
