import { Badge } from "@/components/ui/badge";
import { MATCH_STATUS_LABELS, type EvaluationSessionMatch } from "@/lib/evaluations/client";
import { HumanEvaluationMatchStatus } from "@tcg/backend-core/types/humanEvaluations";

const STATUS_VARIANTS: Record<
  HumanEvaluationMatchStatus,
  "default" | "secondary" | "outline" | "destructive"
> = {
  [HumanEvaluationMatchStatus.SCHEDULED]: "outline",
  [HumanEvaluationMatchStatus.CLAIMED]: "secondary",
  [HumanEvaluationMatchStatus.IN_PROGRESS]: "default",
  [HumanEvaluationMatchStatus.COMPLETED]: "secondary",
  [HumanEvaluationMatchStatus.ABORTED]: "destructive",
};

export function MatchStatusBadge({ status }: { status: HumanEvaluationMatchStatus }) {
  return <Badge variant={STATUS_VARIANTS[status]}>{MATCH_STATUS_LABELS[status]}</Badge>;
}

const OUTCOME_LABELS: Record<NonNullable<EvaluationSessionMatch["outcome"]>, string> = {
  WIN: "You won",
  LOSS: "You lost",
  DRAW: "Draw",
  ABORTED: "Not scored",
};

const OUTCOME_CLASSES: Record<NonNullable<EvaluationSessionMatch["outcome"]>, string> = {
  WIN: "text-emerald-700 border-emerald-600/30 bg-emerald-50",
  LOSS: "text-rose-700 border-rose-600/30 bg-rose-50",
  DRAW: "text-amber-700 border-amber-600/30 bg-amber-50",
  ABORTED: "text-muted-foreground",
};

export function MatchOutcomeBadge({ outcome }: { outcome: EvaluationSessionMatch["outcome"] }) {
  if (outcome === null) {
    return <span className="text-muted-foreground text-sm">—</span>;
  }

  return (
    <Badge variant="outline" className={OUTCOME_CLASSES[outcome]}>
      {OUTCOME_LABELS[outcome]}
    </Badge>
  );
}
