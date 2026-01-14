import { redirect } from "next/navigation";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { DeckList } from "@/components/decks/DeckList";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getUserDecks } from "@tcg/backend-core/services/DeckService";
import type { DeckSummary } from "@tcg/backend-core/types/deck";

export default async function DecksPage() {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  let decks: DeckSummary[];
  let error: string | null = null;

  try {
    decks = await getUserDecks(user.id);
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load decks";
    decks = [];
  }

  return (
    <>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">My Decks</h1>
        <p className="text-muted-foreground mt-2">
          View and manage your card decks
        </p>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <DeckList decks={decks} />
    </>
  );
}
