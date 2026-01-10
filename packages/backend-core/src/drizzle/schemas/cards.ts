import { pgTable, text, integer, pgEnum, unique } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { CardRarity, CardElement, CardType, SpecialCardRarity } from "@/types";
import { citext } from "../customTypes";

export const cardRarityEnum = pgEnum("card_rarity", enumToPgEnum(CardRarity));
export const specialCardRarityEnum = pgEnum("special_card_rarity", enumToPgEnum(SpecialCardRarity));
export const cardElementEnum = pgEnum("card_element", enumToPgEnum(CardElement));
export const cardTypeEnum = pgEnum("card_type", enumToPgEnum(CardType));

export const Cards = pgTable(
  "cards",
  {
    id: uuidv7PrimaryKeyField(),
    cardCode: citext("card_code").notNull(),
    name: text("name").notNull(),
    rarity: cardRarityEnum("rarity").notNull(),
    specialRarity: specialCardRarityEnum("special_rarity"),
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
    imageKey: text("image_url").notNull(),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    unique("cards_card_code_rarity_unique_idx")
      .on(table.cardCode, table.rarity, table.specialRarity)
      .nullsNotDistinct(),
  ]
);
