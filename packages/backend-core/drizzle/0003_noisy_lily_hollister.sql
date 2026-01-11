ALTER TABLE "emails" DROP CONSTRAINT "emails_user_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "decks" DROP CONSTRAINT "decks_user_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "rooms" DROP CONSTRAINT "rooms_player0_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "rooms" DROP CONSTRAINT "rooms_player0_deck_id_decks_id_fk";
--> statement-breakpoint
ALTER TABLE "rooms" DROP CONSTRAINT "rooms_player1_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "rooms" DROP CONSTRAINT "rooms_player1_deck_id_decks_id_fk";
--> statement-breakpoint
ALTER TABLE "match_results" DROP CONSTRAINT "match_results_player0_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "match_results" DROP CONSTRAINT "match_results_player1_id_users_id_fk";
--> statement-breakpoint
ALTER TABLE "game_logs" DROP CONSTRAINT "game_logs_room_id_rooms_id_fk";
--> statement-breakpoint
ALTER TABLE "emails" ADD CONSTRAINT "emails_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "decks" ADD CONSTRAINT "decks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player0_id_users_id_fk" FOREIGN KEY ("player0_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player0_deck_id_decks_id_fk" FOREIGN KEY ("player0_deck_id") REFERENCES "public"."decks"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player1_id_users_id_fk" FOREIGN KEY ("player1_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player1_deck_id_decks_id_fk" FOREIGN KEY ("player1_deck_id") REFERENCES "public"."decks"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_player0_id_users_id_fk" FOREIGN KEY ("player0_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_player1_id_users_id_fk" FOREIGN KEY ("player1_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "game_logs" ADD CONSTRAINT "game_logs_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE cascade ON UPDATE no action;