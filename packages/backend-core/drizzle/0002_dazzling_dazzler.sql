CREATE TYPE "public"."token_type" AS ENUM('ACCESS', 'REFRESH');--> statement-breakpoint
CREATE TABLE "jwt_tokens" (
	"id" uuid PRIMARY KEY NOT NULL,
	"jti" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"token_type" "token_type" NOT NULL,
	"expires_at" timestamp (3) with time zone NOT NULL,
	"revoked_at" timestamp (3) with time zone,
	"created_at" timestamp (3) with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "jwt_tokens_jti_unique" UNIQUE("jti")
);
--> statement-breakpoint
ALTER TABLE "jwt_tokens" ADD CONSTRAINT "jwt_tokens_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "jwt_tokens_user_id_idx" ON "jwt_tokens" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "jwt_tokens_expires_at_idx" ON "jwt_tokens" USING btree ("expires_at");