import { z } from "zod";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getActionTypeName } from "@/lib/game/actionValidation";
import {
  OBSERVATION_KIND_LABELS,
  OBSERVATION_SEVERITY_LABELS,
  OBSERVATION_TAG_LABELS,
} from "@/lib/evaluations/client";
import { HumanEvaluationActorSource } from "@tcg/backend-core/types/humanEvaluations";
import type {
  HumanEvaluationAnnotation,
  HumanEvaluationObservationInput,
  HumanEvaluationReviewAction,
  HumanEvaluationReviewGameLog,
} from "@tcg/backend-core/types/humanEvaluations";

const boardSchema = z.array(z.unknown());

const leaderSchema = z.object({
  cardCode: z.string().nullable(),
  curAtk: z.number(),
  curHp: z.number(),
});

/**
 * Only the fields the timeline renders. Everything else in the persisted
 * observation is still shown verbatim in the raw evidence block.
 */
const observationSchema = z.object({
  phase: z.string(),
  myObservationData: z.object({
    leader: leaderSchema,
    hand: boardSchema,
    garden: boardSchema,
    alley: boardSchema,
    ikzArea: boardSchema,
    deckCount: z.number(),
    ikzPileCount: z.number(),
    hasIkzToken: z.boolean(),
  }),
  opponentObservationData: z.object({
    leader: leaderSchema,
    garden: boardSchema,
    alley: boardSchema,
    ikzArea: boardSchema,
    handCount: z.number(),
    deckCount: z.number(),
    hasIkzToken: z.boolean(),
  }),
});

const actionMaskSchema = z.object({
  legalActionCount: z.number(),
  legalPrimary: z.array(z.number()),
  legalSub1: z.array(z.number()),
  legalSub2: z.array(z.number()),
  legalSub3: z.array(z.number()),
});

interface MatchReviewTimelineProps {
  actions: HumanEvaluationReviewAction[];
  gameLogs: HumanEvaluationReviewGameLog[];
  annotation: HumanEvaluationAnnotation | null;
}

function StateFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-muted-foreground text-[11px]">{label}</dt>
      <dd className="truncate text-xs tabular-nums">{value}</dd>
    </div>
  );
}

function ActionCard({
  action,
  flagged,
}: {
  action: HumanEvaluationReviewAction;
  flagged: HumanEvaluationObservationInput[];
}) {
  const isAi = action.actorSource === HumanEvaluationActorSource.AI;
  const actorLabel = isAi ? "Opponent" : "You";
  const observation = observationSchema.safeParse(action.observation);
  const mask = actionMaskSchema.safeParse(action.legalActionMask);
  const latencyMs = new Date(action.resolvedAt).getTime() - new Date(action.receivedAt).getTime();

  const legalTuples = mask.success
    ? mask.data.legalPrimary.map((primary, index) => ({
        primary,
        sub1: mask.data.legalSub1[index] ?? 0,
        sub2: mask.data.legalSub2[index] ?? 0,
        sub3: mask.data.legalSub3[index] ?? 0,
      }))
    : [];
  const distinctLegalTypes = [...new Set(legalTuples.map((tuple) => tuple.primary))];
  const chosenTupleKey = action.action.join(",");

  return (
    <li className="rounded-lg border">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-4 py-3">
        <span className="text-muted-foreground w-10 shrink-0 text-xs tabular-nums">
          #{action.actionNumber}
        </span>
        <Badge variant={isAi ? "secondary" : "outline"}>
          {actorLabel} · slot {action.actorSlot}
        </Badge>
        <span className="font-mono text-sm">{getActionTypeName(action.action[0])}</span>
        <span className="text-muted-foreground font-mono text-xs">
          [{action.action.join(", ")}]
        </span>
        {action.accepted ? (
          <Badge variant="outline" className="border-emerald-600/30 bg-emerald-50 text-emerald-700">
            Accepted
          </Badge>
        ) : (
          <Badge variant="destructive">Rejected</Badge>
        )}
        <span className="text-muted-foreground ml-auto text-xs tabular-nums">
          {Number.isFinite(latencyMs) ? `${latencyMs} ms` : "—"}
        </span>
      </div>

      <div className="space-y-4 px-4 py-3">
        {action.error ? (
          <p className="text-destructive text-xs">Engine rejection: {action.error}</p>
        ) : null}

        {flagged.length > 0 ? (
          <ul className="flex flex-wrap gap-2">
            {flagged.map((observationEntry, index) => (
              <li key={`${action.actionNumber}-flag-${index}`}>
                <Badge variant="outline" className="border-amber-600/40 bg-amber-50 text-amber-800">
                  {OBSERVATION_KIND_LABELS[observationEntry.kind]} ·{" "}
                  {OBSERVATION_TAG_LABELS[observationEntry.tag]} ·{" "}
                  {OBSERVATION_SEVERITY_LABELS[observationEntry.severity]}
                </Badge>
              </li>
            ))}
          </ul>
        ) : null}

        {observation.success ? (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4 lg:grid-cols-6">
            <StateFact label="Phase" value={observation.data.phase} />
            <StateFact
              label={`${actorLabel} leader`}
              value={`${observation.data.myObservationData.leader.curHp} hp · ${observation.data.myObservationData.leader.curAtk} atk`}
            />
            <StateFact
              label="Their leader"
              value={`${observation.data.opponentObservationData.leader.curHp} hp · ${observation.data.opponentObservationData.leader.curAtk} atk`}
            />
            <StateFact
              label="Hand"
              value={`${observation.data.myObservationData.hand.length} vs ${observation.data.opponentObservationData.handCount}`}
            />
            <StateFact
              label="Board"
              value={`${
                observation.data.myObservationData.garden.filter((slot) => slot !== null).length
              }g/${
                observation.data.myObservationData.alley.filter((slot) => slot !== null).length
              }a vs ${
                observation.data.opponentObservationData.garden.filter((slot) => slot !== null)
                  .length
              }g/${
                observation.data.opponentObservationData.alley.filter((slot) => slot !== null)
                  .length
              }a`}
            />
            <StateFact
              label="Deck · IKZ token"
              value={`${observation.data.myObservationData.deckCount} · ${
                observation.data.myObservationData.hasIkzToken ? "held" : "spent"
              }`}
            />
          </dl>
        ) : (
          <p className="text-muted-foreground text-xs">
            Stored observation does not match the known board shape; see the raw evidence below.
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground text-xs">
            {mask.success
              ? `${mask.data.legalActionCount} legal action${
                  mask.data.legalActionCount === 1 ? "" : "s"
                } available:`
              : "Legal action mask unavailable"}
          </span>
          {distinctLegalTypes.map((primary) => (
            <Badge
              key={primary}
              variant={primary === action.action[0] ? "default" : "outline"}
              className="font-mono text-[11px]"
            >
              {getActionTypeName(primary)}
            </Badge>
          ))}
        </div>

        {legalTuples.length > 0 ? (
          <details>
            <summary className="text-muted-foreground cursor-pointer text-xs">
              All legal alternatives ({legalTuples.length})
            </summary>
            <ul className="mt-2 flex flex-wrap gap-1.5">
              {legalTuples.map((tuple, index) => {
                const tupleKey = [tuple.primary, tuple.sub1, tuple.sub2, tuple.sub3].join(",");

                return (
                  <li key={`${action.actionNumber}-legal-${index}`}>
                    <span
                      className={`inline-flex rounded border px-1.5 py-0.5 font-mono text-[11px] ${
                        tupleKey === chosenTupleKey
                          ? "border-primary bg-primary text-primary-foreground"
                          : "text-muted-foreground"
                      }`}
                    >
                      [{tupleKey.replaceAll(",", ", ")}]
                    </span>
                  </li>
                );
              })}
            </ul>
          </details>
        ) : null}

        <details>
          <summary className="text-muted-foreground cursor-pointer text-xs">
            Raw evidence · state hash{" "}
            <span className="font-mono">{action.stateHash.slice(0, 12)}</span>
          </summary>
          <dl className="mt-2 space-y-2">
            <div>
              <dt className="text-muted-foreground text-[11px]">State hash</dt>
              <dd className="font-mono text-[11px] break-all">{action.stateHash}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground text-[11px]">
                Received {action.receivedAt} · resolved {action.resolvedAt}
              </dt>
              <dd>
                <pre className="bg-muted mt-1 max-h-72 overflow-auto rounded-md p-3 font-mono text-[11px] whitespace-pre-wrap">
                  {JSON.stringify(
                    { observation: action.observation, legalActionMask: action.legalActionMask },
                    null,
                    2
                  )}
                </pre>
              </dd>
            </div>
          </dl>
        </details>
      </div>
    </li>
  );
}

export function MatchReviewTimeline({ actions, gameLogs, annotation }: MatchReviewTimelineProps) {
  const flagsByActionNumber = new Map<number, HumanEvaluationObservationInput[]>();

  for (const observationEntry of annotation?.observations ?? []) {
    if (observationEntry.actionNumber === null) {
      continue;
    }

    const existing = flagsByActionNumber.get(observationEntry.actionNumber) ?? [];
    existing.push(observationEntry);
    flagsByActionNumber.set(observationEntry.actionNumber, existing);
  }

  const logsByBatch = new Map<number, HumanEvaluationReviewGameLog[]>();

  for (const log of gameLogs) {
    const existing = logsByBatch.get(log.batchNumber) ?? [];
    existing.push(log);
    logsByBatch.set(log.batchNumber, existing);
  }

  const orderedBatches = [...logsByBatch.entries()].sort((left, right) => left[0] - right[0]);
  const acceptedCount = actions.filter((action) => action.accepted).length;

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Action timeline</CardTitle>
          <CardDescription>
            Every decision the room accepted or rejected, in order, with the authoritative
            observation and legal action mask the actor was given.{" "}
            {actions.length === 0
              ? "No decisions were recorded for this match."
              : `${actions.length} decisions recorded, ${acceptedCount} accepted.`}
          </CardDescription>
        </CardHeader>
        {actions.length > 0 ? (
          <CardContent>
            <ol className="space-y-3">
              {actions.map((action) => (
                <ActionCard
                  key={`${action.actionNumber}-${action.receivedAt}`}
                  action={action}
                  flagged={flagsByActionNumber.get(action.actionNumber) ?? []}
                />
              ))}
            </ol>
          </CardContent>
        ) : null}
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Engine logs</CardTitle>
          <CardDescription>
            {gameLogs.length === 0
              ? "No engine logs were persisted for this match."
              : `${gameLogs.length} log entries across ${orderedBatches.length} batches, in engine order.`}
          </CardDescription>
        </CardHeader>
        {orderedBatches.length > 0 ? (
          <CardContent className="space-y-2">
            {orderedBatches.map(([batchNumber, logs]) => (
              <details key={batchNumber} className="rounded-md border px-3 py-2">
                <summary className="cursor-pointer text-sm">
                  Batch {batchNumber}
                  <span className="text-muted-foreground ml-2 text-xs">
                    {logs.length} entr{logs.length === 1 ? "y" : "ies"}
                  </span>
                </summary>
                <ol className="mt-3 space-y-2">
                  {logs
                    .slice()
                    .sort((left, right) => left.sequenceNumber - right.sequenceNumber)
                    .map((log) => (
                      <li key={`${batchNumber}-${log.sequenceNumber}`} className="border-l-2 pl-3">
                        <p className="flex flex-wrap items-baseline gap-2 text-xs">
                          <span className="text-muted-foreground tabular-nums">
                            {log.sequenceNumber}
                          </span>
                          <span className="font-mono">{log.logType}</span>
                          <span className="text-muted-foreground">
                            {log.player === null ? "system" : `player ${log.player}`}
                          </span>
                        </p>
                        <pre className="text-muted-foreground mt-1 overflow-x-auto font-mono text-[11px] whitespace-pre-wrap">
                          {JSON.stringify(log.logData)}
                        </pre>
                      </li>
                    ))}
                </ol>
              </details>
            ))}
          </CardContent>
        ) : null}
      </Card>
    </div>
  );
}
