import { pgTable, text, integer, pgEnum } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { CardRarity, CardElement, CardType } from "@/types";

export const cardRarityEnum = pgEnum(
  "card_rarity",
  enumToPgEnum(CardRarity)
);
export const cardElementEnum = pgEnum(
  "card_element",
  enumToPgEnum(CardElement)
);
export const cardTypeEnum = pgEnum("card_type", enumToPgEnum(CardType));

export const cards = pgTable("cards", {
  id: uuidv7PrimaryKeyField(),
  engineId: integer("engine_id").notNull().unique(),
  cardCode: text("card_code").notNull().unique(),
  name: text("name").notNull(),
  rarity: cardRarityEnum("rarity").notNull(),
  element: cardElementEnum("element").notNull(),
  cardType: cardTypeEnum("card_type").notNull(),
  attack: integer("attack"),
  health: integer("health"),
  gatePoints: integer("gate_points"),
  ikzCost: integer("ikz_cost"),
  keywords: text("keywords").array().notNull().default([]),
  subtypes: text("subtypes").array().notNull().default([]),
  effectText: text("effect_text"),
  flavorText: text("flavor_text"),
  imageUrl: text("image_url"),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
