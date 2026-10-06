import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  OBSERVATION_KIND_LABELS,
  OBSERVATION_SEVERITY_LABELS,
  OBSERVATION_TAG_LABELS,
  RATING_DESCRIPTORS,
  RATING_SCALE,
} from "@/lib/evaluations/client";
import type { HumanEvaluationAnnotation } from "@tcg/backend-core/types/humanEvaluations";

interface MatchReviewAnnotationProps {
  annotation: HumanEvaluationAnnotation | null;
  actualModelDisplayName: string;
}

export function MatchReviewAnnotation({
  annotation,
  actualModelDisplayName,
}: MatchReviewAnnotationProps) {
  if (annotation === null) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Your annotation</CardTitle>
          <CardDescription>
            No annotation is stored for this match. Sessions cannot be revealed with a missing
            annotation, so this only happens to data imported outside the normal flow.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your annotation</CardTitle>
        <CardDescription>
          Written before reveal, on {new Date(annotation.createdAt).toLocaleString()}. Frozen.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <ul className="grid gap-x-8 gap-y-4 md:grid-cols-2">
          {RATING_DESCRIPTORS.map((descriptor) => {
            const value = annotation.ratings[descriptor.key];

            return (
              <li key={descriptor.key}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm font-medium">{descriptor.label}</span>
                  <span className="text-sm tabular-nums">{value} / 7</span>
                </div>
                <div className="mt-2 flex gap-1" aria-hidden="true">
                  {RATING_SCALE.map((rating) => (
                    <span
                      key={rating}
                      className={`h-1.5 min-w-0 flex-1 rounded-full ${
                        rating <= value ? "bg-primary" : "bg-secondary"
                      }`}
                    />
                  ))}
                </div>
                <p className="text-muted-foreground mt-1 text-[11px]">
                  {descriptor.lowAnchor} → {descriptor.highAnchor}
                </p>
              </li>
            );
          })}
        </ul>

        <div className="grid gap-4 border-t pt-6 sm:grid-cols-3">
          <div className="min-w-0">
            <p className="text-muted-foreground text-xs">Your blind guess</p>
            <p className="text-sm">{annotation.modelGuess ?? "No guess recorded"}</p>
          </div>
          <div className="min-w-0">
            <p className="text-muted-foreground text-xs">Confidence</p>
            <p className="text-sm tabular-nums">
              {annotation.guessConfidence === null ? "—" : `${annotation.guessConfidence} / 7`}
            </p>
          </div>
          <div className="min-w-0">
            <p className="text-muted-foreground text-xs">Actually played</p>
            <p className="text-sm font-medium">{actualModelDisplayName}</p>
          </div>
        </div>

        <div className="border-t pt-6">
          <p className="text-sm font-medium">
            Tagged moments
            <span className="text-muted-foreground ml-2 text-xs tabular-nums">
              {annotation.observations.length}
            </span>
          </p>
          {annotation.observations.length === 0 ? (
            <p className="text-muted-foreground mt-2 text-xs">
              No specific moments were tagged for this match.
            </p>
          ) : (
            <ul className="mt-3 space-y-2">
              {annotation.observations.map((observation, index) => (
                <li key={index} className="rounded-md border px-3 py-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={observation.kind === "ISSUE" ? "secondary" : "outline"}>
                      {OBSERVATION_KIND_LABELS[observation.kind]}
                    </Badge>
                    <Badge variant="outline">{OBSERVATION_TAG_LABELS[observation.tag]}</Badge>
                    <Badge variant="outline">
                      {OBSERVATION_SEVERITY_LABELS[observation.severity]}
                    </Badge>
                    {observation.actionNumber === null ? null : (
                      <span className="text-muted-foreground text-xs tabular-nums">
                        action #{observation.actionNumber}
                      </span>
                    )}
                  </div>
                  {observation.detail ? <p className="mt-2 text-sm">{observation.detail}</p> : null}
                </li>
              ))}
            </ul>
          )}
        </div>

        {annotation.notes ? (
          <div className="border-t pt-6">
            <p className="text-sm font-medium">Notes</p>
            <p className="mt-2 text-sm whitespace-pre-wrap">{annotation.notes}</p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
