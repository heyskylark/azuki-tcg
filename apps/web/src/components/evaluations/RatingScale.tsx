"use client";

import { RATING_SCALE } from "@/lib/evaluations/client";
import type { HumanEvaluationRating } from "@tcg/backend-core/types/humanEvaluations";

interface RatingScaleProps {
  name: string;
  label: string;
  question: string;
  lowAnchor: string;
  highAnchor: string;
  value: HumanEvaluationRating | null;
  onChange: (value: HumanEvaluationRating) => void;
  disabled?: boolean;
  optional?: boolean;
}

export function RatingScale({
  name,
  label,
  question,
  lowAnchor,
  highAnchor,
  value,
  onChange,
  disabled = false,
  optional = false,
}: RatingScaleProps) {
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className="text-sm font-medium">
        {label}
        {optional ? (
          <span className="text-muted-foreground ml-2 text-xs font-normal">optional</span>
        ) : null}
      </legend>
      <p className="text-muted-foreground mt-1 text-xs">{question}</p>
      <div className="mt-2 flex gap-1">
        {RATING_SCALE.map((rating) => (
          <label key={rating} className="min-w-0 flex-1">
            <input
              type="radio"
              name={name}
              value={rating}
              checked={value === rating}
              onChange={() => onChange(rating)}
              className="peer sr-only"
            />
            <span className="border-input hover:bg-accent peer-checked:border-primary peer-checked:bg-primary peer-checked:text-primary-foreground peer-focus-visible:border-ring peer-focus-visible:ring-ring/50 peer-disabled:pointer-events-none peer-disabled:opacity-50 flex h-9 cursor-pointer items-center justify-center rounded-md border text-sm font-medium tabular-nums transition-colors peer-focus-visible:ring-[3px]">
              {rating}
            </span>
          </label>
        ))}
      </div>
      <div className="text-muted-foreground mt-1.5 flex justify-between text-[11px]">
        <span>1 · {lowAnchor}</span>
        <span>7 · {highAnchor}</span>
      </div>
    </fieldset>
  );
}
