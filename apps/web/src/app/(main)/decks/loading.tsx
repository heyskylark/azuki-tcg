import { DeckListSkeleton } from "@/components/decks/DeckListSkeleton";

export default function DecksLoading() {
  return (
    <>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">My Decks</h1>
        <p className="text-muted-foreground mt-2">
          View and manage your card decks
        </p>
      </div>
      <DeckListSkeleton />
    </>
  );
}
