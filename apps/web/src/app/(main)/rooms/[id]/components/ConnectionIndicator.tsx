"use client";

import { cn } from "@/lib/utils";

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface ConnectionIndicatorProps {
  status: ConnectionStatus;
  className?: string;
}

const statusConfig: Record<ConnectionStatus, { color: string; label: string }> = {
  disconnected: { color: "bg-red-500", label: "Disconnected" },
  connecting: { color: "bg-yellow-500", label: "Connecting" },
  connected: { color: "bg-green-500", label: "Connected" },
  error: { color: "bg-red-500", label: "Error" },
};

export function ConnectionIndicator({ status, className }: ConnectionIndicatorProps) {
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div
        className={cn(
          "h-2.5 w-2.5 rounded-full",
          config.color,
          status === "connecting" && "animate-pulse"
        )}
      />
      <span className="text-sm text-muted-foreground">{config.label}</span>
    </div>
  );
}
