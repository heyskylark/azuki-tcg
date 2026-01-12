CREATE INDEX "rooms_player0_status_idx" ON "rooms" USING btree ("player0_id","status");--> statement-breakpoint
CREATE INDEX "rooms_player1_status_idx" ON "rooms" USING btree ("player1_id","status");