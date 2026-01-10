CREATE TYPE "public"."user_status" AS ENUM('ACTIVE', 'DELETED', 'BANNED');--> statement-breakpoint
CREATE TYPE "public"."user_type" AS ENUM('HUMAN', 'AI');--> statement-breakpoint
CREATE TYPE "public"."card_element" AS ENUM('NORMAL', 'LIGHTNING', 'WATER', 'EARTH', 'FIRE');--> statement-breakpoint
CREATE TYPE "public"."card_rarity" AS ENUM('L', 'L_S', 'G', 'G_S', 'C', 'UC', 'R', 'SR', 'SR_S', 'SR_SS', 'IKZ', 'IKZ_S');--> statement-breakpoint
CREATE TYPE "public"."card_type" AS ENUM('LEADER', 'GATE', 'ENTITY', 'WEAPON', 'SPELL', 'IKZ', 'EXTRA_IKZ');--> statement-breakpoint
CREATE TYPE "public"."special_card_rarity" AS ENUM('INV26_WINNER', 'INV26_SECOND', 'INV26_TOP8', 'TOKEN_AX_WINNER');--> statement-breakpoint
CREATE TYPE "public"."deck_status" AS ENUM('COMPLETE', 'IN_PROGRESS', 'DELETED');--> statement-breakpoint
CREATE TYPE "public"."room_status" AS ENUM('WAITING_FOR_PLAYERS', 'DECK_SELECTION', 'READY_CHECK', 'STARTING', 'IN_MATCH', 'COMPLETED', 'ABORTED');--> statement-breakpoint
CREATE TYPE "public"."room_type" AS ENUM('PRIVATE', 'MATCH_MAKING');--> statement-breakpoint
CREATE TYPE "public"."win_type" AS ENUM('WIN', 'DRAW', 'ABANDON', 'FORFEIT', 'TIMEOUT');--> statement-breakpoint
CREATE TABLE "emails" (
	"id" uuid PRIMARY KEY NOT NULL,
	"email" "citext" NOT NULL,
	"user_id" uuid NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "emails_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY NOT NULL,
	"username" "citext" NOT NULL,
	"password_hash" text NOT NULL,
	"type" "user_type" DEFAULT 'HUMAN' NOT NULL,
	"status" "user_status" DEFAULT 'ACTIVE' NOT NULL,
	"model_key" text,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "users_username_unique" UNIQUE("username")
);
--> statement-breakpoint
CREATE TABLE "cards" (
	"id" uuid PRIMARY KEY NOT NULL,
	"card_code" "citext" NOT NULL,
	"name" text NOT NULL,
	"rarity" "card_rarity" NOT NULL,
	"special_rarity" "special_card_rarity",
	"element" "card_element" NOT NULL,
	"card_type" "card_type" NOT NULL,
	"attack" integer,
	"health" integer,
	"gate_points" integer,
	"ikz_cost" integer,
	"keywords" text[] DEFAULT '{}' NOT NULL,
	"subtypes" text[] DEFAULT '{}' NOT NULL,
	"effect_text" text,
	"flavor_text" text,
	"image_url" text NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "cards_card_code_rarity_unique_idx" UNIQUE NULLS NOT DISTINCT("card_code","rarity","special_rarity")
);
--> statement-breakpoint
CREATE TABLE "decks" (
	"id" uuid PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"user_id" uuid NOT NULL,
	"status" "deck_status" DEFAULT 'IN_PROGRESS' NOT NULL,
	"is_system_deck" boolean DEFAULT false NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "system_deck_not_deleted" CHECK (NOT ("decks"."is_system_deck" = true AND "decks"."status" = 'DELETED'))
);
--> statement-breakpoint
CREATE TABLE "deck_card_junctions" (
	"id" uuid PRIMARY KEY NOT NULL,
	"deck_id" uuid NOT NULL,
	"card_id" uuid NOT NULL,
	"quantity" integer DEFAULT 1 NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "rooms" (
	"id" uuid PRIMARY KEY NOT NULL,
	"status" "room_status" DEFAULT 'WAITING_FOR_PLAYERS' NOT NULL,
	"type" "room_type" DEFAULT 'PRIVATE' NOT NULL,
	"password_hash" text,
	"world_id" text,
	"rng_seed" integer,
	"player0_id" uuid,
	"player0_deck_id" uuid,
	"player0_ready" boolean DEFAULT false NOT NULL,
	"player1_id" uuid,
	"player1_deck_id" uuid,
	"player1_ready" boolean DEFAULT false NOT NULL,
	"deck_selection_deadline" timestamp with time zone,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "match_results" (
	"id" uuid PRIMARY KEY NOT NULL,
	"room_id" uuid NOT NULL,
	"player0_id" uuid NOT NULL,
	"player1_id" uuid NOT NULL,
	"winner_id" uuid,
	"win_type" "win_type" NOT NULL,
	"total_turns" integer NOT NULL,
	"duration_seconds" integer NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "game_logs" (
	"id" uuid PRIMARY KEY NOT NULL,
	"room_id" uuid NOT NULL,
	"batch_number" integer NOT NULL,
	"sequence_number" integer NOT NULL,
	"log_type" text NOT NULL,
	"player" integer,
	"log_data" jsonb NOT NULL,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "emails" ADD CONSTRAINT "emails_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "decks" ADD CONSTRAINT "decks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "deck_card_junctions" ADD CONSTRAINT "deck_card_junctions_deck_id_decks_id_fk" FOREIGN KEY ("deck_id") REFERENCES "public"."decks"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "deck_card_junctions" ADD CONSTRAINT "deck_card_junctions_card_id_cards_id_fk" FOREIGN KEY ("card_id") REFERENCES "public"."cards"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player0_id_users_id_fk" FOREIGN KEY ("player0_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player0_deck_id_decks_id_fk" FOREIGN KEY ("player0_deck_id") REFERENCES "public"."decks"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player1_id_users_id_fk" FOREIGN KEY ("player1_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "rooms" ADD CONSTRAINT "rooms_player1_deck_id_decks_id_fk" FOREIGN KEY ("player1_deck_id") REFERENCES "public"."decks"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_player0_id_users_id_fk" FOREIGN KEY ("player0_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_player1_id_users_id_fk" FOREIGN KEY ("player1_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "match_results" ADD CONSTRAINT "match_results_winner_id_users_id_fk" FOREIGN KEY ("winner_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "game_logs" ADD CONSTRAINT "game_logs_room_id_rooms_id_fk" FOREIGN KEY ("room_id") REFERENCES "public"."rooms"("id") ON DELETE no action ON UPDATE no action;