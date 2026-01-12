import { Suspense } from "react";
import { notFound, redirect } from "next/navigation";
import { getServerUser } from "@/lib/auth/getServerUser";
import { findRoomById } from "@tcg/backend-core/services/roomService";
import { RoomClient, type RoomData } from "./RoomClient";
import { RoomSkeleton } from "./RoomSkeleton";

interface PageProps {
  params: Promise<{ id: string }>;
}

async function RoomContent({ roomId }: { roomId: string }) {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const room = await findRoomById(roomId);

  if (!room || !room.player0Id) {
    notFound();
  }

  const roomData: RoomData = {
    id: room.id,
    status: room.status,
    type: room.type,
    hasPassword: room.passwordHash !== null,
    player0Id: room.player0Id,
    player1Id: room.player1Id,
    createdAt: room.createdAt.toISOString(),
    updatedAt: room.updatedAt.toISOString(),
  };

  return <RoomClient initialRoom={roomData} user={user} />;
}

export default async function RoomPage({ params }: PageProps) {
  const { id: roomId } = await params;

  return (
    <Suspense fallback={<RoomSkeleton />}>
      <RoomContent roomId={roomId} />
    </Suspense>
  );
}
