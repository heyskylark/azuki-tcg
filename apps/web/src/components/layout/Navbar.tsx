"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";

export function Navbar() {
  const { user, logout } = useAuth();

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
