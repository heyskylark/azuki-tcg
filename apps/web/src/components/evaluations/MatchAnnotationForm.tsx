"use client";

import { useCallback, useRef, useState } from "react";
import { Plus, X } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { RatingScale } from "@/components/evaluations/RatingScale";
import {
  OBSERVATION_KINDS,
  OBSERVATION_KIND_LABELS,
  OBSERVATION_SEVERITIES,
  OBSERVATION_SEVERITY_LABELS,
  OBSERVATION_TAGS,
  OBSERVATION_TAG_LABELS,
  RATING_DESCRIPTORS,
  submitEvaluationAnnotation,
  type EvaluationAnnotation,
  type EvaluationObservationKind,
  type EvaluationObservationSeverity,
  type EvaluationObservationTag,
  type EvaluationRatings,
} from "@/lib/evaluations/client";
import type {
  HumanEvaluationAnnotationInput,
  HumanEvaluationRating,
} from "@tcg/backend-core/types/humanEvaluations";

interface ObservationDraft {
  key: number;
  kind: EvaluationObservationKind;
  tag: EvaluationObservationTag;
  severity: EvaluationObservationSeverity;
  actionNumber: string;
  detail: string;
}

interface MatchAnnotationFormProps {
  matchId: string;
  opponentLabel: string;
  submitLabel?: string;
  onSubmitted: (annotation: EvaluationAnnotation) => void;
  onCancel?: () => void;
}

const NATIVE_SELECT_CLASS =
  "border-input focus-visible:border-ring focus-visible:ring-ring/50 h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50";

function SegmentedChoice<TOption extends string>({
  name,
  legend,
  options,
  labels,
  value,
  onChange,
  disabled,
}: {
  name: string;
  legend: string;
  options: readonly TOption[];
  labels: Record<TOption, string>;
  value: TOption;
  onChange: (next: TOption) => void;
  disabled: boolean;
}) {
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className="text-muted-foreground mb-1.5 text-xs font-medium">{legend}</legend>
      <div className="flex gap-1">
        {options.map((option) => (
          <label key={option} className="min-w-0 flex-1">
            <input
              type="radio"
              name={name}
              value={option}
              checked={value === option}
              onChange={() => onChange(option)}
              className="peer sr-only"
            />
            <span className="border-input hover:bg-accent peer-checked:border-primary peer-checked:bg-primary peer-checked:text-primary-foreground peer-focus-visible:border-ring peer-focus-visible:ring-ring/50 peer-disabled:pointer-events-none peer-disabled:opacity-50 flex h-9 cursor-pointer items-center justify-center truncate rounded-md border px-2 text-xs font-medium transition-colors peer-focus-visible:ring-[3px]">
              {labels[option]}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function MatchAnnotationForm({
  matchId,
  opponentLabel,
  submitLabel = "Save annotation",
  onSubmitted,
  onCancel,
}: MatchAnnotationFormProps) {
  const nextObservationKey = useRef(1);
  const [ratings, setRatings] = useState<Map<keyof EvaluationRatings, HumanEvaluationRating>>(
    new Map()
  );
  const [modelGuess, setModelGuess] = useState("");
  const [guessConfidence, setGuessConfidence] = useState<HumanEvaluationRating | null>(null);
  const [observations, setObservations] = useState<ObservationDraft[]>([]);
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const updateObservation = useCallback(
    (key: number, patch: Partial<Omit<ObservationDraft, "key">>) => {
      setObservations((current) =>
        current.map((draft) => (draft.key === key ? { ...draft, ...patch } : draft))
      );
    },
    []
  );

  const handleAddObservation = useCallback(() => {
    const key = nextObservationKey.current;
    nextObservationKey.current += 1;
    setObservations((current) => [
      ...current,
      {
        key,
        kind: "ISSUE",
        tag: "MISPLAY",
        severity: "MODERATE",
        actionNumber: "",
        detail: "",
      },
    ]);
  }, []);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    const opponentStrength = ratings.get("opponentStrength");
    const decisionQuality = ratings.get("decisionQuality");
    const deckCoherence = ratings.get("deckCoherence");
    const humanLikeness = ratings.get("humanLikeness");
    const matchEnjoyment = ratings.get("matchEnjoyment");

    if (
      opponentStrength === undefined ||
      decisionQuality === undefined ||
      deckCoherence === undefined ||
      humanLikeness === undefined ||
      matchEnjoyment === undefined
    ) {
      setError("Rate all five dimensions before saving. Every played match needs a full rating.");
      return;
    }

    const payload: HumanEvaluationAnnotationInput = {
      ratings: {
        opponentStrength,
        decisionQuality,
        deckCoherence,
        humanLikeness,
        matchEnjoyment,
      },
      modelGuess: modelGuess.trim() === "" ? null : modelGuess.trim(),
      guessConfidence,
      observations: observations.map((draft) => {
        const parsedActionNumber = Number.parseInt(draft.actionNumber, 10);

        return {
          kind: draft.kind,
          tag: draft.tag,
          severity: draft.severity,
          actionNumber:
            Number.isInteger(parsedActionNumber) && parsedActionNumber > 0
              ? parsedActionNumber
              : null,
          detail: draft.detail.trim() === "" ? null : draft.detail.trim(),
        };
      }),
      notes: notes.trim() === "" ? null : notes.trim(),
    };

    setIsSaving(true);

    try {
      const annotation = await submitEvaluationAnnotation(matchId, payload);
      onSubmitted(annotation);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Failed to save annotation");
      setIsSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid gap-x-8 gap-y-5 md:grid-cols-2">
        {RATING_DESCRIPTORS.map((descriptor) => (
          <RatingScale
            key={descriptor.key}
            name={`${matchId}-${descriptor.key}`}
            label={descriptor.label}
            question={descriptor.question}
            lowAnchor={descriptor.lowAnchor}
            highAnchor={descriptor.highAnchor}
            value={ratings.get(descriptor.key) ?? null}
            onChange={(value) =>
              setRatings((current) => new Map(current).set(descriptor.key, value))
            }
            disabled={isSaving}
          />
        ))}
      </div>

      <div className="grid gap-x-8 gap-y-5 border-t pt-5 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor={`${matchId}-model-guess`}>Who do you think this was?</Label>
          <Input
            id={`${matchId}-model-guess`}
            value={modelGuess}
            onChange={(event) => setModelGuess(event.target.value)}
            placeholder="Free text — a training stage, a behaviour, or “no idea”"
            disabled={isSaving}
            autoComplete="off"
          />
          <p className="text-muted-foreground text-xs">
            Recorded before reveal so your guesses stay honest. {opponentLabel} is a stable label
            inside this session only.
          </p>
        </div>
        <RatingScale
          name={`${matchId}-guess-confidence`}
          label="Confidence in that guess"
          question="How sure are you?"
          lowAnchor="Pure coin flip"
          highAnchor="Certain"
          value={guessConfidence}
          onChange={setGuessConfidence}
          disabled={isSaving}
          optional
        />
      </div>

      <div className="space-y-3 border-t pt-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium">Issues and missed opportunities</h3>
            <p className="text-muted-foreground mt-1 text-xs">
              Tag each concrete moment. Reference the action number from the frozen review timeline
              when you can — it is the same numbering the reviewer sees after reveal.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleAddObservation}
            disabled={isSaving}
          >
            <Plus />
            Add observation
          </Button>
        </div>

        {observations.length === 0 ? (
          <p className="text-muted-foreground rounded-md border border-dashed px-4 py-6 text-center text-xs">
            No tagged moments yet. Ratings alone are a valid annotation — add observations only for
            behaviour worth going back to.
          </p>
        ) : (
          <ul className="space-y-3">
            {observations.map((draft, index) => (
              <li key={draft.key} className="rounded-lg border p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <span className="text-xs font-medium">Observation {index + 1}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    onClick={() =>
                      setObservations((current) =>
                        current.filter((entry) => entry.key !== draft.key)
                      )
                    }
                    disabled={isSaving}
                    aria-label={`Remove observation ${index + 1}`}
                  >
                    <X />
                  </Button>
                </div>
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_9rem]">
                  <SegmentedChoice
                    name={`${matchId}-obs-${draft.key}-kind`}
                    legend="Kind"
                    options={OBSERVATION_KINDS}
                    labels={OBSERVATION_KIND_LABELS}
                    value={draft.kind}
                    onChange={(kind) => updateObservation(draft.key, { kind })}
                    disabled={isSaving}
                  />
                  <div className="min-w-0">
                    <Label
                      htmlFor={`${matchId}-obs-${draft.key}-tag`}
                      className="text-muted-foreground mb-1.5 text-xs font-medium"
                    >
                      Tag
                    </Label>
                    <select
                      id={`${matchId}-obs-${draft.key}-tag`}
                      value={draft.tag}
                      onChange={(event) => {
                        const nextTag = OBSERVATION_TAGS.find((tag) => tag === event.target.value);
                        if (nextTag) {
                          updateObservation(draft.key, { tag: nextTag });
                        }
                      }}
                      disabled={isSaving}
                      className={NATIVE_SELECT_CLASS}
                    >
                      {OBSERVATION_TAGS.map((tag) => (
                        <option key={tag} value={tag}>
                          {OBSERVATION_TAG_LABELS[tag]}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="min-w-0">
                    <Label
                      htmlFor={`${matchId}-obs-${draft.key}-action`}
                      className="text-muted-foreground mb-1.5 text-xs font-medium"
                    >
                      Action #
                    </Label>
                    <Input
                      id={`${matchId}-obs-${draft.key}-action`}
                      type="number"
                      min={1}
                      step={1}
                      inputMode="numeric"
                      value={draft.actionNumber}
                      onChange={(event) =>
                        updateObservation(draft.key, { actionNumber: event.target.value })
                      }
                      placeholder="—"
                      disabled={isSaving}
                    />
                  </div>
                </div>
                <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
                  <SegmentedChoice
                    name={`${matchId}-obs-${draft.key}-severity`}
                    legend="Severity"
                    options={OBSERVATION_SEVERITIES}
                    labels={OBSERVATION_SEVERITY_LABELS}
                    value={draft.severity}
                    onChange={(severity) => updateObservation(draft.key, { severity })}
                    disabled={isSaving}
                  />
                  <div className="min-w-0">
                    <Label
                      htmlFor={`${matchId}-obs-${draft.key}-detail`}
                      className="text-muted-foreground mb-1.5 text-xs font-medium"
                    >
                      What happened
                    </Label>
                    <Textarea
                      id={`${matchId}-obs-${draft.key}-detail`}
                      value={draft.detail}
                      onChange={(event) =>
                        updateObservation(draft.key, { detail: event.target.value })
                      }
                      rows={2}
                      placeholder="Attacked into an obvious defender and lost its only flyer."
                      disabled={isSaving}
                    />
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="space-y-2 border-t pt-5">
        <Label htmlFor={`${matchId}-notes`}>Notes</Label>
        <Textarea
          id={`${matchId}-notes`}
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          rows={4}
          placeholder="Anything the ratings miss: how the game was decided, what the deck was trying to do, whether you were tired."
          disabled={isSaving}
        />
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={isSaving}>
          {isSaving ? "Saving..." : submitLabel}
        </Button>
        {onCancel ? (
          <Button type="button" variant="ghost" onClick={onCancel} disabled={isSaving}>
            Cancel
          </Button>
        ) : null}
        <p className="text-muted-foreground text-xs">
          Annotations are final once saved and are stored before any identity is revealed.
        </p>
      </div>
    </form>
  );
}
