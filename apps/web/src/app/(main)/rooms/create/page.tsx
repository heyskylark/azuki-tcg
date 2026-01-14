import { Suspense } from "react";
import { redirect } from "next/navigation";
import { getServerUser } from "@/lib/auth/getServerUser";
import { findActiveRoomForUser } from "@tcg/backend-core/services/roomService";
import { CreateRoomSkeleton } from "./CreateRoomSkeleton";
import { ActiveRoomCard } from "./ActiveRoomCard";
import { CreateRoomForm } from "./CreateRoomForm";

async function CreateRoomContent() {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const activeRoom = await findActiveRoomForUser(user.id);

  if (activeRoom) {
    return <ActiveRoomCard roomId={activeRoom.id} />;
  }

  return <CreateRoomForm />;
}

export default function CreateRoomPage() {
  return (
    <Suspense fallback={<CreateRoomSkeleton />}>
      <CreateRoomContent />
    </Suspense>
  );
}
