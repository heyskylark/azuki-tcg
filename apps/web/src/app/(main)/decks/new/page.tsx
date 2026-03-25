import { redirect } from "next/navigation";
import { DeckBuilderForm } from "@/components/decks/DeckBuilderForm";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getDeckBuilderCards } from "@tcg/backend-core/services/DeckService";

export default async function NewDeckPage() {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const availableCards = await getDeckBuilderCards();

  return <DeckBuilderForm availableCards={availableCards} mode="create" />;
}
