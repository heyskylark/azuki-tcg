"use client";

import { Button } from "@/components/ui/button";

interface GameOverScreenProps {
  outcome: "win" | "lose" | "draw";
  reason?: string | null;
  onReturnHome: () => void;
}

export function GameOverScreen({
  outcome,
  reason,
  onReturnHome,
}: GameOverScreenProps) {
  const isWin = outcome === "win";
  const isLose = outcome === "lose";
  const title = isWin ? "You won" : isLose ? "You lose" : "Game over";
  const accentClass = isWin ? "text-emerald-300" : isLose ? "text-rose-300" : "text-white";

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-black px-8 py-10 text-center text-white shadow-2xl">
        <div className={`text-4xl font-bold ${accentClass}`}>{title}</div>
        {reason ? (
          <p className="mt-3 text-sm text-white/70">{reason}</p>
        ) : null}
        <Button className="mt-8 w-full" onClick={onReturnHome}>
          Back to Home
        </Button>
      </div>
    </div>
  );
}
