ALTER TYPE "public"."room_status" ADD VALUE 'CLOSED';--> statement-breakpoint
ALTER TYPE "public"."token_type" ADD VALUE 'JOIN';
COMMIT;
BEGIN;
