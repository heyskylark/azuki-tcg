import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { cardCodeToDefId, defIdToCardCode } from "@core/services/cardMapperService";

const CARD_DEFS_HEADER = path.resolve(
  import.meta.dirname,
  "../../../../include/generated/card_defs.h"
);

function readEngineCardDefIds(): Array<{ cardCode: string; defId: number }> {
  const header = readFileSync(CARD_DEFS_HEADER, "utf8");
  return Array.from(header.matchAll(/CARD_DEF_([A-Z]+\d*)_(\d+) = (\d+),/g), (match) => ({
    cardCode: `${match[1]}-${match[2]}`,
    defId: Number(match[3]),
  }));
}

test("every engine CardDefId maps to its card code in both directions", () => {
  const engineIds = readEngineCardDefIds();
  assert.ok(engineIds.length > 0, "no CardDefIds parsed from card_defs.h");

  for (const { cardCode, defId } of engineIds) {
    assert.equal(cardCodeToDefId(cardCode), defId, `${cardCode} -> defId`);
    assert.equal(defIdToCardCode(defId), cardCode, `${defId} -> cardCode`);
  }
});
