import { createHash, randomBytes } from "node:crypto";
import { and, asc, count, eq, inArray, isNull, max, or } from "drizzle-orm";
import { uuidv7 } from "uuidv7";
import db, { type IDatabase, type ITransaction } from "@core/database";
import {
  AiModels,
  Cards,
  DeckCardJunctions,
  Decks,
  GameLogs,
  HumanEvaluationActions,
  HumanEvaluationAnnotationObservations,
  HumanEvaluationAnnotations,
  HumanEvaluationDeckArtifacts,
  HumanEvaluationMatches,
  HumanEvaluationSessions,
  MatchResults,
  Rooms,
  Users,
} from "@core/drizzle/schemas";
import {
  HumanEvaluationBlindError,
  HumanEvaluationNotFoundError,
  HumanEvaluationUnavailableError,
  ValidationError,
} from "@core/errors";
import {
  AiModelStatus,
  DeckStatus,
  RoomStatus,
  RoomType,
  UserStatus,
  UserType,
  WinType,
} from "@core/types";
import { CardElement, CardRarity, CardType, RarityOrdering } from "@core/types/cards";
import {
  HumanEvaluationActorSource,
  HumanEvaluationMatchStatus,
  HumanEvaluationSessionStatus,
  type HumanEvaluationAnnotationInput,
  type HumanEvaluationAnnotation,
  type HumanEvaluationDeckArtifactInput,
  type HumanEvaluationRuntimeMatch,
  type HumanEvaluationReview,
  type HumanEvaluationSessionSummary,
} from "@core/types/humanEvaluations";
import { lockUserRoomMembership } from "@core/services/roomService";

type Database = IDatabase | ITransaction;
const PROTOCOL = "human-eval-v1";
const IKZ_CARD_CODE = "IKZ-001";
const ACTIVE_ROOM_STATUSES = [
  RoomStatus.WAITING_FOR_PLAYERS,
  RoomStatus.DECK_SELECTION,
  RoomStatus.READY_CHECK,
  RoomStatus.STARTING,
  RoomStatus.IN_MATCH,
];

const EVALUATION_CONTEXTS = [
  { gateCardCode: "AZK01-120", leaderCardCode: "AZK01-119" },
  { gateCardCode: "AZK01-120", leaderCardCode: "STT01-001" },
  { gateCardCode: "STT01-002", leaderCardCode: "AZK01-119" },
  { gateCardCode: "STT01-002", leaderCardCode: "STT01-001" },
  { gateCardCode: "AZK01-122", leaderCardCode: "AZK01-121" },
  { gateCardCode: "AZK01-122", leaderCardCode: "STT04-001" },
  { gateCardCode: "STT04-002", leaderCardCode: "AZK01-121" },
  { gateCardCode: "STT04-002", leaderCardCode: "STT04-001" },
  { gateCardCode: "AZK01-124", leaderCardCode: "AZK01-123" },
  { gateCardCode: "AZK01-124", leaderCardCode: "STT03-001" },
  { gateCardCode: "STT03-002", leaderCardCode: "AZK01-123" },
  { gateCardCode: "STT03-002", leaderCardCode: "STT03-001" },
  { gateCardCode: "AZK01-126", leaderCardCode: "AZK01-125" },
  { gateCardCode: "AZK01-126", leaderCardCode: "STT02-001" },
  { gateCardCode: "STT02-002", leaderCardCode: "AZK01-125" },
  { gateCardCode: "STT02-002", leaderCardCode: "STT02-001" },
];

function deterministicHex(seed: string, label: string): string {
  return createHash("sha256").update(`${PROTOCOL}:${seed}:${label}`).digest("hex");
}

function deterministicInt(seed: string, label: string): number {
  return Number.parseInt(deterministicHex(seed, label).slice(0, 8), 16) & 0x7fffffff;
}

export function createHumanEvaluationRuntimeSessionKey(): string {
  return `human-eval:${randomBytes(32).toString("hex")}`;
}

function deterministicOrder<T>(values: T[], seed: string, label: string): T[] {
  return values
    .map((value, index) => ({ value, key: deterministicHex(seed, `${label}:${index}`) }))
    .sort((left, right) => left.key.localeCompare(right.key))
    .map(({ value }) => value);
}

function asSlot(value: number): 0 | 1 {
  if (value === 0 || value === 1) return value;
  throw new Error("Invalid evaluation player slot");
}

async function requireOwnedSession(sessionId: string, reviewerId: string, database: Database) {
  const session = await database
    .select()
    .from(HumanEvaluationSessions)
    .where(
      and(
        eq(HumanEvaluationSessions.id, sessionId),
        eq(HumanEvaluationSessions.reviewerId, reviewerId)
      )
    )
    .limit(1)
    .then((rows) => rows[0]);
  if (!session) throw new HumanEvaluationNotFoundError();
  return session;
}

function isFinishedStatus(status: HumanEvaluationMatchStatus): boolean {
  return (
    status === HumanEvaluationMatchStatus.COMPLETED || status === HumanEvaluationMatchStatus.ABORTED
  );
}

export function canMaterializeHumanEvaluationDeck(status: HumanEvaluationMatchStatus): boolean {
  return (
    status === HumanEvaluationMatchStatus.CLAIMED ||
    status === HumanEvaluationMatchStatus.IN_PROGRESS
  );
}
async function requireOwnedMatch(matchId: string, reviewerId: string, database: Database) {
  const row = await database
    .select({ match: HumanEvaluationMatches, session: HumanEvaluationSessions })
    .from(HumanEvaluationMatches)
    .innerJoin(
      HumanEvaluationSessions,
      eq(HumanEvaluationMatches.sessionId, HumanEvaluationSessions.id)
    )
    .where(
      and(
        eq(HumanEvaluationMatches.id, matchId),
        eq(HumanEvaluationSessions.reviewerId, reviewerId)
      )
    )
    .limit(1)
    .then((rows) => rows[0]);
  if (!row) throw new HumanEvaluationNotFoundError();
  return row;
}

interface HumanDeckSnapshotSource {
  name: string;
  junctions: Array<{ cardId: string; quantity: number }>;
}

export function copyHumanDeckSnapshotJunctions(
  snapshotDeckId: string,
  junctions: ReadonlyArray<{ cardId: string; quantity: number }>
): Array<{ deckId: string; cardId: string; quantity: number }> {
  return junctions.map(({ cardId, quantity }) => ({
    deckId: snapshotDeckId,
    cardId,
    quantity,
  }));
}

async function validateHumanDeck(
  reviewerId: string,
  deckId: string,
  database: Database
): Promise<HumanDeckSnapshotSource> {
  const deck = await database
    .select({
      name: Decks.name,
      status: Decks.status,
      evaluation: Decks.isEvaluationGenerated,
    })
    .from(Decks)
    .where(and(eq(Decks.id, deckId), eq(Decks.userId, reviewerId)))
    .limit(1)
    .for("update")
    .then((rows) => rows[0]);
  if (!deck || deck.status !== DeckStatus.COMPLETE || deck.evaluation) {
    throw new ValidationError("Select a complete deck that you own");
  }
  const cards = await database
    .select({
      cardId: DeckCardJunctions.cardId,
      cardType: Cards.cardType,
      quantity: DeckCardJunctions.quantity,
    })
    .from(DeckCardJunctions)
    .innerJoin(Cards, eq(DeckCardJunctions.cardId, Cards.id))
    .where(eq(DeckCardJunctions.deckId, deckId));
  let leaders = 0;
  let gates = 0;
  let main = 0;
  for (const card of cards) {
    if (card.cardType === CardType.LEADER) leaders += card.quantity;
    else if (card.cardType === CardType.GATE) gates += card.quantity;
    else if (
      card.cardType === CardType.ENTITY ||
      card.cardType === CardType.SPELL ||
      card.cardType === CardType.WEAPON
    ) {
      main += card.quantity;
    }
  }
  if (leaders !== 1 || gates !== 1 || main !== 50) {
    throw new ValidationError("The selected deck is not battle legal");
  }
  return {
    name: deck.name,
    junctions: cards.map(({ cardId, quantity }) => ({ cardId, quantity })),
  };
}

function buildEvaluationAiUsername(matchId: string, modelKey: string): string {
  const identityHash = createHash("sha256").update(`${matchId}:${modelKey}`).digest("hex");
  return `ai_eval_${identityHash.slice(0, 32)}`;
}

async function getOrCreateEvaluationAiUser(
  matchId: string,
  modelKey: string,
  database: Database
): Promise<string> {
  const username = buildEvaluationAiUsername(matchId, modelKey);
  const existing = await database
    .select({ id: Users.id })
    .from(Users)
    .where(and(eq(Users.type, UserType.AI), eq(Users.username, username)))
    .limit(1)
    .then((rows) => rows[0]);
  if (existing) return existing.id;
  const created = await database
    .insert(Users)
    .values({
      username,
      displayName: "AI Opponent",
      passwordHash: uuidv7(),
      type: UserType.AI,
      status: UserStatus.ACTIVE,
      modelKey,
    })
    .onConflictDoNothing()
    .returning({ id: Users.id })
    .then((rows) => rows[0]);
  if (created) return created.id;
  const raced = await database
    .select({ id: Users.id })
    .from(Users)
    .where(and(eq(Users.type, UserType.AI), eq(Users.username, username)))
    .limit(1)
    .then((rows) => rows[0]);
  if (!raced) throw new Error("Failed to resolve evaluation AI user");
  return raced.id;
}

export async function createHumanEvaluationSession(
  input: { reviewerId: string; humanDeckId: string; gamesPerModel: 8 | 16 },
  database: IDatabase = db
): Promise<HumanEvaluationSessionSummary> {
  const activeSession = await database
    .select({ id: HumanEvaluationSessions.id })
    .from(HumanEvaluationSessions)
    .where(
      and(
        eq(HumanEvaluationSessions.reviewerId, input.reviewerId),
        eq(HumanEvaluationSessions.status, HumanEvaluationSessionStatus.ACTIVE)
      )
    )
    .limit(1)
    .then((rows) => rows[0]);
  if (activeSession) {
    throw new HumanEvaluationUnavailableError(
      "Resume or reveal your active evaluation before creating another"
    );
  }
  const models = await database
    .select()
    .from(AiModels)
    .where(
      and(eq(AiModels.status, AiModelStatus.ENABLED), eq(AiModels.humanEvaluationEnabled, true))
    )
    .orderBy(asc(AiModels.id));
  if (models.length === 0)
    throw new HumanEvaluationUnavailableError("No models are enabled for human evaluation");
  if (models.some((model) => model.checkpointSha256.trim().length === 0)) {
    throw new HumanEvaluationUnavailableError(
      "An enabled evaluation model is missing its checkpoint fingerprint"
    );
  }
  const seed = randomBytes(32).toString("hex");
  const sessionId = uuidv7();
  await database.transaction(async (tx) => {
    const sourceDeck = await validateHumanDeck(input.reviewerId, input.humanDeckId, tx);
    const snapshotDeck = await tx
      .insert(Decks)
      .values({
        name: `Evaluation snapshot: ${sourceDeck.name}`,
        userId: input.reviewerId,
        status: DeckStatus.COMPLETE,
        isSystemDeck: false,
        isEvaluationGenerated: true,
      })
      .returning({ id: Decks.id })
      .then((rows) => rows[0]);
    if (!snapshotDeck) {
      throw new Error("Failed to snapshot the human evaluation deck");
    }
    await tx
      .insert(DeckCardJunctions)
      .values(copyHumanDeckSnapshotJunctions(snapshotDeck.id, sourceDeck.junctions));
    await tx.insert(HumanEvaluationSessions).values({
      id: sessionId,
      reviewerId: input.reviewerId,
      humanDeckId: snapshotDeck.id,
      sourceHumanDeckId: input.humanDeckId,
      gamesPerModel: input.gamesPerModel,
      protocol: PROTOCOL,
      scheduleSeed: seed,
    });
    const scheduled: Array<typeof HumanEvaluationMatches.$inferInsert> = [];
    for (const [modelIndex, model] of models.entries()) {
      const allContexts = deterministicOrder(
        [...EVALUATION_CONTEXTS],
        seed,
        `contexts:${model.id}`
      );
      const contexts =
        input.gamesPerModel === 16
          ? allContexts
          : deterministicOrder(
              allContexts.filter(
                (context, index, array) =>
                  array.findIndex(
                    (candidate) => candidate.gateCardCode === context.gateCardCode
                  ) === index
              ),
              seed,
              `eight-contexts:${model.id}`
            );
      const cells = deterministicOrder(
        Array.from({ length: input.gamesPerModel }, (_, index) => ({
          aiSlot: asSlot((index % 4) >> 1),
          startingPlayer: asSlot(index % 2),
        })),
        seed,
        `cells:${model.id}`
      );
      for (let gameIndex = 0; gameIndex < input.gamesPerModel; gameIndex += 1) {
        const context = contexts[gameIndex];
        const cell = cells[gameIndex];
        if (!context || !cell) throw new Error("Failed to build evaluation schedule");
        const matchId = uuidv7();
        scheduled.push({
          id: matchId,
          sessionId,
          ordinal: modelIndex * input.gamesPerModel + gameIndex + 1,
          modelId: model.id,
          modelKeySnapshot: model.modelKey,
          modelDisplayNameSnapshot: model.displayName,
          checkpointSha256Snapshot: model.checkpointSha256,
          sessionKey: createHumanEvaluationRuntimeSessionKey(),
          draftSeed: deterministicInt(seed, `draft:${matchId}`),
          battleSeed: deterministicInt(seed, `battle:${matchId}`),
          aiSlot: cell.aiSlot,
          startingPlayer: cell.startingPlayer,
          gateCardCode: context.gateCardCode,
          leaderCardCode: context.leaderCardCode,
        });
      }
    }
    const shuffled = deterministicOrder(scheduled, seed, "schedule").map((match, index) => ({
      ...match,
      ordinal: index + 1,
    }));
    await tx.insert(HumanEvaluationMatches).values(shuffled);
  });
  return getHumanEvaluationSession(sessionId, input.reviewerId, database);
}

function outcomeForMatch(
  status: HumanEvaluationMatchStatus,
  winnerId: string | null | undefined,
  reviewerId: string,
  winType: WinType | null | undefined
): "WIN" | "LOSS" | "DRAW" | "ABORTED" | null {
  if (status === HumanEvaluationMatchStatus.ABORTED) return "ABORTED";
  if (status !== HumanEvaluationMatchStatus.COMPLETED) return null;
  if (winType === WinType.DRAW || winnerId == null) return "DRAW";
  return winnerId === reviewerId ? "WIN" : "LOSS";
}

export async function getHumanEvaluationSession(
  sessionId: string,
  reviewerId: string,
  database: Database = db
): Promise<HumanEvaluationSessionSummary> {
  const session = await requireOwnedSession(sessionId, reviewerId, database);
  const rows = await database
    .select({
      match: HumanEvaluationMatches,
      result: MatchResults,
      annotationId: HumanEvaluationAnnotations.id,
    })
    .from(HumanEvaluationMatches)
    .leftJoin(MatchResults, eq(HumanEvaluationMatches.matchResultId, MatchResults.id))
    .leftJoin(
      HumanEvaluationAnnotations,
      eq(HumanEvaluationMatches.id, HumanEvaluationAnnotations.matchId)
    )
    .where(eq(HumanEvaluationMatches.sessionId, sessionId))
    .orderBy(asc(HumanEvaluationMatches.ordinal));
  const finished = rows.filter(({ match }) => isFinishedStatus(match.status));
  const annotatedMatches = rows.filter(({ annotationId }) => annotationId !== null).length;
  const revealed = session.revealedAt !== null;
  return {
    id: session.id,
    humanDeckId: session.sourceHumanDeckId,
    gamesPerModel: session.gamesPerModel === 8 ? 8 : 16,
    totalMatches: rows.length,
    completedMatches: finished.length,
    annotatedMatches,
    revealReady:
      rows.length > 0 && finished.length === rows.length && annotatedMatches === rows.length,
    createdAt: session.createdAt.toISOString(),
    revealedAt: session.revealedAt?.toISOString() ?? null,
    matches: rows.map(({ match, result, annotationId }) => ({
      matchId: match.id,
      ordinal: match.ordinal,
      status: match.status,
      opponentLabel: `Opponent ${match.ordinal}`,
      roomId: match.roomId,
      outcome: outcomeForMatch(match.status, result?.winnerId, reviewerId, result?.winType),
      hasAnnotation: annotationId !== null,
      revealed: revealed
        ? {
            modelDisplayName: match.modelDisplayNameSnapshot,
            checkpointSha256: match.checkpointSha256Snapshot,
          }
        : null,
    })),
  };
}

export async function listHumanEvaluationSessionsForReviewer(
  reviewerId: string,
  database: Database = db
): Promise<HumanEvaluationSessionSummary[]> {
  const sessions = await database
    .select({ id: HumanEvaluationSessions.id })
    .from(HumanEvaluationSessions)
    .where(eq(HumanEvaluationSessions.reviewerId, reviewerId))
    .orderBy(asc(HumanEvaluationSessions.createdAt));
  return Promise.all(sessions.map(({ id }) => getHumanEvaluationSession(id, reviewerId, database)));
}

export async function claimNextHumanEvaluationMatch(
  sessionId: string,
  reviewerId: string,
  database: IDatabase = db
): Promise<{
  roomId: string;
  matchId: string;
  ordinal: number;
  totalMatches: number;
  resumed: boolean;
}> {
  return database.transaction(async (tx) => {
    const session = await tx
      .select()
      .from(HumanEvaluationSessions)
      .where(
        and(
          eq(HumanEvaluationSessions.id, sessionId),
          eq(HumanEvaluationSessions.reviewerId, reviewerId)
        )
      )
      .limit(1)
      .for("update")
      .then((rows) => rows[0]);
    if (!session) {
      throw new HumanEvaluationNotFoundError();
    }
    if (session.revealedAt) {
      throw new HumanEvaluationUnavailableError("This evaluation is already revealed");
    }
    await lockUserRoomMembership(reviewerId, tx);
    const totalMatches = await tx
      .select({ value: count() })
      .from(HumanEvaluationMatches)
      .where(eq(HumanEvaluationMatches.sessionId, sessionId))
      .then((rows) => Number(rows[0]?.value ?? 0));
    const existing = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(
        and(
          eq(HumanEvaluationMatches.sessionId, sessionId),
          inArray(HumanEvaluationMatches.status, [
            HumanEvaluationMatchStatus.CLAIMED,
            HumanEvaluationMatchStatus.IN_PROGRESS,
          ])
        )
      )
      .orderBy(asc(HumanEvaluationMatches.ordinal))
      .limit(1)
      .then((rows) => rows[0]);
    if (existing?.roomId) {
      return {
        roomId: existing.roomId,
        matchId: existing.id,
        ordinal: existing.ordinal,
        totalMatches,
        resumed: true,
      };
    }
    const unannotatedFinishedMatch = await tx
      .select({ id: HumanEvaluationMatches.id })
      .from(HumanEvaluationMatches)
      .leftJoin(
        HumanEvaluationAnnotations,
        eq(HumanEvaluationMatches.id, HumanEvaluationAnnotations.matchId)
      )
      .where(
        and(
          eq(HumanEvaluationMatches.sessionId, sessionId),
          inArray(HumanEvaluationMatches.status, [
            HumanEvaluationMatchStatus.COMPLETED,
            HumanEvaluationMatchStatus.ABORTED,
          ]),
          isNull(HumanEvaluationAnnotations.id)
        )
      )
      .limit(1)
      .then((rows) => rows[0]);
    if (unannotatedFinishedMatch) {
      throw new HumanEvaluationUnavailableError(
        "Annotate the finished match before starting the next opponent"
      );
    }
    const activeRoom = await tx
      .select({ id: Rooms.id })
      .from(Rooms)
      .where(
        and(
          inArray(Rooms.status, ACTIVE_ROOM_STATUSES),
          or(eq(Rooms.player0Id, reviewerId), eq(Rooms.player1Id, reviewerId))
        )
      )
      .limit(1)
      .then((rows) => rows[0]);
    if (activeRoom) {
      throw new HumanEvaluationUnavailableError("Finish or close your active room first");
    }
    const match = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(
        and(
          eq(HumanEvaluationMatches.sessionId, sessionId),
          eq(HumanEvaluationMatches.status, HumanEvaluationMatchStatus.SCHEDULED)
        )
      )
      .orderBy(asc(HumanEvaluationMatches.ordinal))
      .limit(1)
      .for("update", { skipLocked: true })
      .then((rows) => rows[0]);
    if (!match) {
      throw new HumanEvaluationUnavailableError("No evaluation matches remain");
    }
    const aiUserId = await getOrCreateEvaluationAiUser(match.id, match.modelKeySnapshot, tx);
    const aiSlot = asSlot(match.aiSlot);
    const humanSlot = asSlot(1 - aiSlot);
    const room = await tx
      .insert(Rooms)
      .values({
        status: RoomStatus.DECK_SELECTION,
        type: RoomType.PRIVATE,
        rngSeed: match.battleSeed,
        aiModelId: match.modelId,
        player0Id: humanSlot === 0 ? reviewerId : aiUserId,
        player1Id: humanSlot === 1 ? reviewerId : aiUserId,
        player0DeckId: humanSlot === 0 ? session.humanDeckId : null,
        player1DeckId: humanSlot === 1 ? session.humanDeckId : null,
        player0Ready: humanSlot === 0,
        player1Ready: humanSlot === 1,
      })
      .returning({ id: Rooms.id })
      .then((rows) => rows[0]);
    if (!room) {
      throw new Error("Failed to materialize evaluation room");
    }
    await tx
      .update(HumanEvaluationMatches)
      .set({
        roomId: room.id,
        status: HumanEvaluationMatchStatus.CLAIMED,
        claimedAt: new Date(),
      })
      .where(
        and(
          eq(HumanEvaluationMatches.id, match.id),
          eq(HumanEvaluationMatches.status, HumanEvaluationMatchStatus.SCHEDULED)
        )
      );
    return {
      roomId: room.id,
      matchId: match.id,
      ordinal: match.ordinal,
      totalMatches,
      resumed: false,
    };
  });
}

export async function getHumanEvaluationMatch(
  matchId: string,
  reviewerId: string,
  database: Database = db
) {
  const { match } = await requireOwnedMatch(matchId, reviewerId, database);
  const totalMatches = await database
    .select({ value: count() })
    .from(HumanEvaluationMatches)
    .where(eq(HumanEvaluationMatches.sessionId, match.sessionId))
    .then((rows) => Number(rows[0]?.value ?? 0));
  const annotation = await database
    .select({ id: HumanEvaluationAnnotations.id })
    .from(HumanEvaluationAnnotations)
    .where(eq(HumanEvaluationAnnotations.matchId, matchId))
    .limit(1)
    .then((rows) => rows[0]);
  const result = match.matchResultId
    ? await database
        .select()
        .from(MatchResults)
        .where(eq(MatchResults.id, match.matchResultId))
        .limit(1)
        .then((rows) => rows[0])
    : null;
  return {
    matchId: match.id,
    sessionId: match.sessionId,
    ordinal: match.ordinal,
    totalMatches,
    status: match.status,
    opponentLabel: `Opponent ${match.ordinal}`,
    roomId: match.roomId,
    outcome: outcomeForMatch(match.status, result?.winnerId, reviewerId, result?.winType),
    hasAnnotation: Boolean(annotation),
  };
}

export async function getHumanEvaluationMatchByRoomId(
  roomId: string,
  database: Database = db
): Promise<HumanEvaluationRuntimeMatch | null> {
  const row = await database
    .select({ match: HumanEvaluationMatches, deckId: HumanEvaluationDeckArtifacts.deckId })
    .from(HumanEvaluationMatches)
    .leftJoin(
      HumanEvaluationDeckArtifacts,
      eq(HumanEvaluationMatches.id, HumanEvaluationDeckArtifacts.matchId)
    )
    .where(eq(HumanEvaluationMatches.roomId, roomId))
    .limit(1)
    .then((rows) => rows[0]);
  if (!row?.match.roomId) return null;
  const totalMatches = await database
    .select({ value: count() })
    .from(HumanEvaluationMatches)
    .where(eq(HumanEvaluationMatches.sessionId, row.match.sessionId))
    .then((rows) => Number(rows[0]?.value ?? 0));
  const aiSlot = asSlot(row.match.aiSlot);
  return {
    matchId: row.match.id,
    sessionId: row.match.sessionId,
    ordinal: row.match.ordinal,
    totalMatches,
    roomId: row.match.roomId,
    modelKey: row.match.modelKeySnapshot,
    checkpointSha256: row.match.checkpointSha256Snapshot,
    sessionKey: row.match.sessionKey,
    draftSeed: row.match.draftSeed,
    battleSeed: row.match.battleSeed,
    aiSlot,
    humanSlot: asSlot(1 - aiSlot),
    startingPlayer: asSlot(row.match.startingPlayer),
    gateCardCode: row.match.gateCardCode,
    leaderCardCode: row.match.leaderCardCode,
    generatedDeckId: row.deckId ?? null,
    status: row.match.status,
  };
}

function validateArtifact(
  match: typeof HumanEvaluationMatches.$inferSelect,
  artifact: HumanEvaluationDeckArtifactInput
): void {
  if (
    artifact.checkpointSha256 !== match.checkpointSha256Snapshot ||
    artifact.gateCardCode !== match.gateCardCode ||
    artifact.leaderCardCode !== match.leaderCardCode
  ) {
    throw new HumanEvaluationUnavailableError(
      "Generated deck does not match its frozen assignment"
    );
  }
  if (artifact.orderedMainCardCodes.length !== 50)
    throw new ValidationError("Generated deck must contain exactly 50 ordered main cards");
  const counts = new Map<string, number>();
  for (const code of artifact.orderedMainCardCodes) counts.set(code, (counts.get(code) ?? 0) + 1);
  if (
    Object.keys(artifact.cardCounts).length !== counts.size ||
    [...counts].some(([code, quantity]) => artifact.cardCounts[code] !== quantity)
  ) {
    throw new ValidationError("Generated deck counts do not match its ordered cards");
  }
  if (
    artifact.picks.length !== 50 ||
    artifact.picks.some(
      (pick, index) =>
        pick.ordinal !== index + 1 ||
        pick.candidateCardCodes[pick.selectedIndex] !== pick.selectedCardCode ||
        pick.selectedCardCode !== artifact.orderedMainCardCodes[index]
    )
  ) {
    throw new ValidationError("Generated deck draft evidence is inconsistent");
  }
}

function serializeArtifactEvidence(artifact: HumanEvaluationDeckArtifactInput): string {
  return JSON.stringify([
    artifact.gateCardCode,
    artifact.leaderCardCode,
    artifact.orderedMainCardCodes,
    Object.entries(artifact.cardCounts).sort(([leftCode], [rightCode]) =>
      leftCode.localeCompare(rightCode)
    ),
    artifact.picks.map((pick) => [
      pick.ordinal,
      pick.candidateCardCodes,
      pick.selectedIndex,
      pick.selectedCardCode,
    ]),
    artifact.deckHash,
    artifact.catalogHash,
    artifact.checkpointSha256,
  ]);
}

export async function materializeHumanEvaluationDeck(
  matchId: string,
  artifact: HumanEvaluationDeckArtifactInput,
  database: IDatabase = db
): Promise<{ deckId: string }> {
  return database.transaction(async (tx) => {
    const match = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(eq(HumanEvaluationMatches.id, matchId))
      .limit(1)
      .for("update")
      .then((rows) => rows[0]);
    if (!match?.roomId) throw new HumanEvaluationNotFoundError();
    if (!canMaterializeHumanEvaluationDeck(match.status)) {
      throw new HumanEvaluationUnavailableError(
        "Evaluation match cannot materialize a deck after finalization"
      );
    }
    validateArtifact(match, artifact);
    const existing = await tx
      .select()
      .from(HumanEvaluationDeckArtifacts)
      .where(eq(HumanEvaluationDeckArtifacts.matchId, matchId))
      .limit(1)
      .then((rows) => rows[0]);
    if (existing) {
      if (serializeArtifactEvidence(existing) !== serializeArtifactEvidence(artifact)) {
        throw new HumanEvaluationUnavailableError(
          "A different generated deck is already frozen for this match"
        );
      }
      return { deckId: existing.deckId };
    }
    const room = await tx
      .select()
      .from(Rooms)
      .where(eq(Rooms.id, match.roomId))
      .limit(1)
      .then((rows) => rows[0]);
    if (!room) throw new HumanEvaluationNotFoundError();
    const aiUserId = match.aiSlot === 0 ? room.player0Id : room.player1Id;
    if (!aiUserId) throw new Error("Evaluation room has no AI user");
    const allCodes = [
      ...new Set([
        artifact.gateCardCode,
        artifact.leaderCardCode,
        ...artifact.orderedMainCardCodes,
        IKZ_CARD_CODE,
      ]),
    ];
    const candidates = await tx
      .select({
        id: Cards.id,
        cardCode: Cards.cardCode,
        rarity: Cards.rarity,
        cardType: Cards.cardType,
        element: Cards.element,
      })
      .from(Cards)
      .where(and(inArray(Cards.cardCode, allCodes), isNull(Cards.specialRarity)));
    const selected = new Map<
      string,
      { id: string; rarity: CardRarity; cardType: CardType; element: CardElement }
    >();
    for (const card of candidates) {
      const current = selected.get(card.cardCode);
      if (!current || RarityOrdering[card.rarity] < RarityOrdering[current.rarity])
        selected.set(card.cardCode, card);
    }
    if (selected.size !== allCodes.length)
      throw new ValidationError("Generated deck references unknown cards");
    const gateCard = selected.get(artifact.gateCardCode);
    const leaderCard = selected.get(artifact.leaderCardCode);
    const ikzCard = selected.get(IKZ_CARD_CODE);
    if (
      gateCard?.cardType !== CardType.GATE ||
      leaderCard?.cardType !== CardType.LEADER ||
      ikzCard?.cardType !== CardType.IKZ ||
      (leaderCard.element !== CardElement.NORMAL && leaderCard.element !== gateCard.element) ||
      artifact.orderedMainCardCodes.some((code) => {
        const card = selected.get(code);
        return (
          (card?.cardType !== CardType.ENTITY &&
            card?.cardType !== CardType.SPELL &&
            card?.cardType !== CardType.WEAPON) ||
          (card.element !== CardElement.NORMAL && card.element !== gateCard.element)
        );
      })
    ) {
      throw new ValidationError("Generated deck contains cards in invalid deck zones");
    }
    const deck = await tx
      .insert(Decks)
      .values({
        name: `Evaluation Opponent ${match.ordinal}`,
        userId: aiUserId,
        status: DeckStatus.COMPLETE,
        isSystemDeck: false,
        isEvaluationGenerated: true,
      })
      .returning({ id: Decks.id })
      .then((rows) => rows[0]);
    if (!deck) throw new Error("Failed to persist generated evaluation deck");
    const quantities = new Map<string, number>(Object.entries(artifact.cardCounts));
    quantities.set(artifact.gateCardCode, (quantities.get(artifact.gateCardCode) ?? 0) + 1);
    quantities.set(artifact.leaderCardCode, (quantities.get(artifact.leaderCardCode) ?? 0) + 1);
    quantities.set(IKZ_CARD_CODE, 10);
    await tx.insert(DeckCardJunctions).values(
      [...quantities].map(([code, quantity]) => {
        const card = selected.get(code);
        if (!card) throw new ValidationError(`Unknown generated card ${code}`);
        return { deckId: deck.id, cardId: card.id, quantity };
      })
    );
    await tx.insert(HumanEvaluationDeckArtifacts).values({ matchId, deckId: deck.id, ...artifact });
    const roomDeckUpdate =
      match.aiSlot === 0
        ? { player0DeckId: deck.id, player0Ready: true }
        : { player1DeckId: deck.id, player1Ready: true };
    await tx.update(Rooms).set(roomDeckUpdate).where(eq(Rooms.id, match.roomId));
    await tx
      .update(HumanEvaluationMatches)
      .set({ status: HumanEvaluationMatchStatus.IN_PROGRESS })
      .where(eq(HumanEvaluationMatches.id, matchId));
    return { deckId: deck.id };
  });
}

export interface RecordHumanEvaluationActionInput {
  roomId: string;
  actorSlot: 0 | 1;
  actorSource: HumanEvaluationActorSource | "HUMAN" | "AI";
  action: [number, number, number, number];
  accepted: boolean;
  error: string | null;
  observation: unknown;
  legalActionMask: unknown;
  stateHash: string;
  receivedAt: Date;
  resolvedAt: Date;
}

export async function recordHumanEvaluationAction(
  input: RecordHumanEvaluationActionInput,
  database: IDatabase = db
): Promise<{ actionNumber: number }> {
  return database.transaction(async (tx) => {
    const match = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(eq(HumanEvaluationMatches.roomId, input.roomId))
      .limit(1)
      .for("update")
      .then((rows) => rows[0]);
    if (!match) throw new HumanEvaluationNotFoundError();
    if (
      match.status !== HumanEvaluationMatchStatus.CLAIMED &&
      match.status !== HumanEvaluationMatchStatus.IN_PROGRESS
    ) {
      throw new HumanEvaluationUnavailableError("Evaluation match is not accepting actions");
    }
    const expectedSource =
      input.actorSlot === match.aiSlot
        ? HumanEvaluationActorSource.AI
        : HumanEvaluationActorSource.HUMAN;
    if (input.actorSource !== expectedSource) {
      throw new ValidationError("Evaluation action actor source does not match its player slot");
    }
    if (input.resolvedAt < input.receivedAt) {
      throw new ValidationError("Evaluation action resolution precedes receipt");
    }
    const current = await tx
      .select({ value: max(HumanEvaluationActions.actionNumber) })
      .from(HumanEvaluationActions)
      .where(eq(HumanEvaluationActions.matchId, match.id))
      .then((rows) => rows[0]?.value ?? 0);
    const actionNumber = current + 1;
    await tx
      .insert(HumanEvaluationActions)
      .values({ matchId: match.id, actionNumber, ...input, actorSource: expectedSource });
    return { actionNumber };
  });
}

export interface FinalizeHumanEvaluationMatchInput {
  roomId: string;
  winnerId: string | null;
  winType: WinType;
  totalTurns: number;
  durationSeconds: number;
}

export async function finalizeHumanEvaluationMatch(
  input: FinalizeHumanEvaluationMatchInput,
  database: IDatabase = db
): Promise<{ matchResultId: string }> {
  return database.transaction(async (tx) => {
    const match = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(eq(HumanEvaluationMatches.roomId, input.roomId))
      .limit(1)
      .for("update")
      .then((rows) => rows[0]);
    if (!match) throw new HumanEvaluationNotFoundError();
    if (match.matchResultId) {
      const existing = await tx
        .select()
        .from(MatchResults)
        .where(eq(MatchResults.id, match.matchResultId))
        .limit(1)
        .then((rows) => rows[0]);
      if (
        !existing ||
        existing.roomId !== input.roomId ||
        existing.winnerId !== input.winnerId ||
        existing.winType !== input.winType ||
        existing.totalTurns !== input.totalTurns ||
        existing.durationSeconds !== input.durationSeconds
      ) {
        throw new HumanEvaluationUnavailableError(
          "A different result is already frozen for this evaluation match"
        );
      }
      return { matchResultId: match.matchResultId };
    }
    if (isFinishedStatus(match.status)) {
      throw new HumanEvaluationUnavailableError("Evaluation match is already finalized");
    }
    const room = await tx
      .select()
      .from(Rooms)
      .where(eq(Rooms.id, input.roomId))
      .limit(1)
      .then((rows) => rows[0]);
    if (!room?.player0Id || !room.player1Id) throw new Error("Evaluation room is missing players");
    if (
      input.winnerId !== null &&
      input.winnerId !== room.player0Id &&
      input.winnerId !== room.player1Id
    ) {
      throw new ValidationError("Evaluation winner is not a room player");
    }
    if (input.totalTurns < 0 || input.durationSeconds < 0) {
      throw new ValidationError("Evaluation result metrics cannot be negative");
    }
    const result = await tx
      .insert(MatchResults)
      .values({
        roomId: input.roomId,
        player0Id: room.player0Id,
        player1Id: room.player1Id,
        aiModelId: match.modelId,
        winnerId: input.winnerId,
        winType: input.winType,
        totalTurns: input.totalTurns,
        durationSeconds: input.durationSeconds,
      })
      .returning({ id: MatchResults.id })
      .then((rows) => rows[0]);
    if (!result) throw new Error("Failed to persist evaluation result");
    await tx
      .update(HumanEvaluationMatches)
      .set({
        matchResultId: result.id,
        status: HumanEvaluationMatchStatus.COMPLETED,
        completedAt: new Date(),
      })
      .where(eq(HumanEvaluationMatches.id, match.id));
    await tx.update(Rooms).set({ status: RoomStatus.COMPLETED }).where(eq(Rooms.id, input.roomId));
    return { matchResultId: result.id };
  });
}

export async function abortHumanEvaluationRoom(
  roomId: string,
  database: IDatabase = db
): Promise<void> {
  await database.transaction(async (tx) => {
    const match = await tx
      .select()
      .from(HumanEvaluationMatches)
      .where(eq(HumanEvaluationMatches.roomId, roomId))
      .limit(1)
      .for("update")
      .then((rows) => rows[0]);
    if (!match) return;
    if (isFinishedStatus(match.status)) return;
    await tx
      .update(HumanEvaluationMatches)
      .set({ status: HumanEvaluationMatchStatus.ABORTED, completedAt: new Date() })
      .where(eq(HumanEvaluationMatches.id, match.id));
    await tx.update(Rooms).set({ status: RoomStatus.ABORTED }).where(eq(Rooms.id, roomId));
  });
}

function annotationResponse(
  annotation: typeof HumanEvaluationAnnotations.$inferSelect,
  observations: Array<typeof HumanEvaluationAnnotationObservations.$inferSelect>
): HumanEvaluationAnnotation {
  return {
    matchId: annotation.matchId,
    ratings: {
      opponentStrength: annotation.opponentStrengthRating,
      decisionQuality: annotation.decisionQualityRating,
      deckCoherence: annotation.deckCoherenceRating,
      humanLikeness: annotation.humanLikenessRating,
      matchEnjoyment: annotation.matchEnjoymentRating,
    },
    modelGuess: annotation.modelGuess,
    guessConfidence: annotation.guessConfidence,
    observations: observations.map(({ kind, tag, severity, actionNumber, detail }) => ({
      kind,
      tag,
      severity,
      actionNumber,
      detail,
    })),
    notes: annotation.notes,
    createdAt: annotation.createdAt.toISOString(),
  };
}

export async function annotateHumanEvaluationMatch(
  matchId: string,
  reviewerId: string,
  input: HumanEvaluationAnnotationInput,
  database: IDatabase = db
) {
  return database.transaction(async (tx) => {
    const { match, session } = await requireOwnedMatch(matchId, reviewerId, tx);
    if (session.revealedAt)
      throw new HumanEvaluationUnavailableError("Annotations are frozen after reveal");
    if (!isFinishedStatus(match.status)) {
      throw new HumanEvaluationUnavailableError("Finish this match before annotating it");
    }
    const existing = await tx
      .select()
      .from(HumanEvaluationAnnotations)
      .where(eq(HumanEvaluationAnnotations.matchId, matchId))
      .limit(1)
      .then((rows) => rows[0]);
    if (existing)
      throw new HumanEvaluationUnavailableError("This match already has its frozen annotation");
    const referenced = input.observations
      .map((observation) => observation.actionNumber)
      .filter((value): value is number => value !== null);
    if (referenced.length > 0) {
      const found = await tx
        .select({ actionNumber: HumanEvaluationActions.actionNumber })
        .from(HumanEvaluationActions)
        .where(
          and(
            eq(HumanEvaluationActions.matchId, matchId),
            inArray(HumanEvaluationActions.actionNumber, referenced)
          )
        );
      if (new Set(found.map(({ actionNumber }) => actionNumber)).size !== new Set(referenced).size)
        throw new ValidationError("An observation references an unknown action");
    }
    const annotation = await tx
      .insert(HumanEvaluationAnnotations)
      .values({
        matchId,
        reviewerId,
        opponentStrengthRating: input.ratings.opponentStrength,
        decisionQualityRating: input.ratings.decisionQuality,
        deckCoherenceRating: input.ratings.deckCoherence,
        humanLikenessRating: input.ratings.humanLikeness,
        matchEnjoymentRating: input.ratings.matchEnjoyment,
        modelGuess: input.modelGuess,
        guessConfidence: input.guessConfidence,
        notes: input.notes,
      })
      .returning()
      .then((rows) => rows[0]);
    if (!annotation) throw new Error("Failed to persist evaluation annotation");
    const observations =
      input.observations.length === 0
        ? []
        : await tx
            .insert(HumanEvaluationAnnotationObservations)
            .values(
              input.observations.map((observation, ordinal) => ({
                annotationId: annotation.id,
                ordinal,
                ...observation,
              }))
            )
            .returning();
    return annotationResponse(annotation, observations);
  });
}

export async function revealHumanEvaluationSession(
  sessionId: string,
  reviewerId: string,
  database: IDatabase = db
): Promise<HumanEvaluationSessionSummary> {
  await database.transaction(async (tx) => {
    const session = await requireOwnedSession(sessionId, reviewerId, tx);
    if (session.revealedAt) return;
    const matches = await tx
      .select({
        status: HumanEvaluationMatches.status,
        annotationId: HumanEvaluationAnnotations.id,
      })
      .from(HumanEvaluationMatches)
      .leftJoin(
        HumanEvaluationAnnotations,
        eq(HumanEvaluationMatches.id, HumanEvaluationAnnotations.matchId)
      )
      .where(eq(HumanEvaluationMatches.sessionId, sessionId));
    if (
      matches.length === 0 ||
      matches.some(({ status, annotationId }) => !isFinishedStatus(status) || annotationId === null)
    ) {
      throw new HumanEvaluationUnavailableError(
        "Every scheduled match must be complete and annotated before reveal"
      );
    }
    await tx
      .update(HumanEvaluationSessions)
      .set({ status: HumanEvaluationSessionStatus.REVEALED, revealedAt: new Date() })
      .where(
        and(eq(HumanEvaluationSessions.id, sessionId), isNull(HumanEvaluationSessions.revealedAt))
      );
  });
  return getHumanEvaluationSession(sessionId, reviewerId, database);
}

export async function getHumanEvaluationReview(
  matchId: string,
  reviewerId: string,
  database: Database = db
): Promise<HumanEvaluationReview> {
  const { match, session } = await requireOwnedMatch(matchId, reviewerId, database);
  if (!session.revealedAt) throw new HumanEvaluationBlindError();
  const artifact = await database
    .select()
    .from(HumanEvaluationDeckArtifacts)
    .where(eq(HumanEvaluationDeckArtifacts.matchId, matchId))
    .limit(1)
    .then((rows) => rows[0]);
  const actions = await database
    .select()
    .from(HumanEvaluationActions)
    .where(eq(HumanEvaluationActions.matchId, matchId))
    .orderBy(asc(HumanEvaluationActions.actionNumber));
  const logs = match.roomId
    ? await database
        .select()
        .from(GameLogs)
        .where(eq(GameLogs.roomId, match.roomId))
        .orderBy(asc(GameLogs.batchNumber), asc(GameLogs.sequenceNumber))
    : [];
  const result = match.matchResultId
    ? await database
        .select()
        .from(MatchResults)
        .where(eq(MatchResults.id, match.matchResultId))
        .limit(1)
        .then((rows) => rows[0])
    : null;
  const annotation = await database
    .select()
    .from(HumanEvaluationAnnotations)
    .where(eq(HumanEvaluationAnnotations.matchId, matchId))
    .limit(1)
    .then((rows) => rows[0]);
  const observations = annotation
    ? await database
        .select()
        .from(HumanEvaluationAnnotationObservations)
        .where(eq(HumanEvaluationAnnotationObservations.annotationId, annotation.id))
        .orderBy(asc(HumanEvaluationAnnotationObservations.ordinal))
    : [];
  return {
    matchId,
    sessionId: session.id,
    ordinal: match.ordinal,
    model: {
      displayName: match.modelDisplayNameSnapshot,
      modelKey: match.modelKeySnapshot,
      checkpointSha256: match.checkpointSha256Snapshot,
    },
    assignment: {
      aiSlot: match.aiSlot,
      startingPlayer: match.startingPlayer,
      gateCardCode: match.gateCardCode,
      leaderCardCode: match.leaderCardCode,
      battleSeed: match.battleSeed,
    },
    draft: artifact
      ? {
          orderedMainCardCodes: artifact.orderedMainCardCodes,
          cardCounts: artifact.cardCounts,
          picks: artifact.picks,
          deckHash: artifact.deckHash,
          catalogHash: artifact.catalogHash,
          checkpointSha256: artifact.checkpointSha256,
        }
      : null,
    actions: actions.map(
      ({
        actionNumber,
        actorSlot,
        actorSource,
        action,
        accepted,
        error,
        observation,
        legalActionMask,
        stateHash,
        receivedAt,
        resolvedAt,
      }) => ({
        actionNumber,
        actorSlot,
        actorSource,
        action,
        accepted,
        error,
        observation,
        legalActionMask,
        stateHash,
        receivedAt: receivedAt.toISOString(),
        resolvedAt: resolvedAt.toISOString(),
      })
    ),
    gameLogs: logs.map(({ batchNumber, sequenceNumber, logType, player, logData, createdAt }) => ({
      batchNumber,
      sequenceNumber,
      logType,
      player,
      logData,
      createdAt: createdAt.toISOString(),
    })),
    result: result
      ? {
          winnerId: result.winnerId,
          winType: result.winType,
          totalTurns: result.totalTurns,
          durationSeconds: result.durationSeconds,
        }
      : null,
    annotation: annotation ? annotationResponse(annotation, observations) : null,
  };
}
