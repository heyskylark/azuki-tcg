CREATE TYPE "public"."special_card_rarity" AS ENUM('INV26_WINNER', 'INV26_SECOND', 'INV26_TOP8', 'TOKEN_AX_WINNER');--> statement-breakpoint
ALTER TYPE "public"."card_rarity" ADD VALUE 'L_S' BEFORE 'G';--> statement-breakpoint
ALTER TYPE "public"."card_rarity" ADD VALUE 'G_S' BEFORE 'C';--> statement-breakpoint
ALTER TYPE "public"."card_rarity" ADD VALUE 'SR_S' BEFORE 'IKZ';--> statement-breakpoint
ALTER TYPE "public"."card_rarity" ADD VALUE 'SR_SS' BEFORE 'IKZ';--> statement-breakpoint
ALTER TYPE "public"."card_rarity" ADD VALUE 'IKZ_S';--> statement-breakpoint
ALTER TABLE "cards" DROP CONSTRAINT "cards_engine_id_unique";--> statement-breakpoint
ALTER TABLE "cards" DROP CONSTRAINT "cards_card_code_unique";--> statement-breakpoint
ALTER TABLE "decks" DROP CONSTRAINT "decks_leader_card_id_cards_id_fk";
--> statement-breakpoint
ALTER TABLE "decks" DROP CONSTRAINT "decks_gate_card_id_cards_id_fk";
--> statement-breakpoint
ALTER TABLE "cards" ALTER COLUMN "image_url" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "cards" ADD COLUMN "special_rarity" "special_card_rarity";--> statement-breakpoint
ALTER TABLE "cards" DROP COLUMN "engine_id";--> statement-breakpoint
ALTER TABLE "decks" DROP COLUMN "leader_card_id";--> statement-breakpoint
ALTER TABLE "decks" DROP COLUMN "gate_card_id";--> statement-breakpoint
ALTER TABLE "cards" ADD CONSTRAINT "cards_card_code_rarity_unique_idx" UNIQUE("card_code","rarity","special_rarity");