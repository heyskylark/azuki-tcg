"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";
import { useRoom } from "@/contexts/RoomContext";
import { Button } from "@/components/ui/button";

function formatRoomStatus(status: string): string {
  switch (status) {
    case "WAITING_FOR_PLAYERS":
      return "Waiting";
    case "DECK_SELECTION":
      return "Deck Selection";
    case "READY_CHECK":
      return "Ready Check";
    case "STARTING":
      return "Starting";
    case "IN_MATCH":
      return "In Match";
    default:
      return status;
  }
}

export function Navbar() {
  const { user, logout } = useAuth();
  const { activeRoom, roomState, connectionStatus } = useRoom();

  return (
    <nav className="border-b bg-background">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <Link href="/dashboard" className="text-xl font-bold">
            Azuki TCG
          </Link>
          <div className="hidden md:flex items-center space-x-4">
            <Link
              href="/dashboard"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Dashboard
            </Link>
            <Link
              href="/decks"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Decks
            </Link>
            <Link
              href="/rooms/create"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Create Room
            </Link>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          {activeRoom && (
            <Link
              href={`/rooms/${activeRoom.id}`}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-primary/10 hover:bg-primary/20 transition-colors"
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  connectionStatus === "connected"
                    ? "bg-green-500"
                    : connectionStatus === "connecting"
                      ? "bg-yellow-500 animate-pulse"
                      : "bg-red-500"
                }`}
              />
              <span className="text-sm font-medium">
                {roomState ? formatRoomStatus(roomState.status) : "Room"}
              </span>
            </Link>
          )}
          <Link
            href="/profile"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            {user?.displayName}
          </Link>
          <Button variant="outline" size="sm" onClick={logout}>
            Logout
          </Button>
        </div>
      </div>
    </nav>
  );
}
