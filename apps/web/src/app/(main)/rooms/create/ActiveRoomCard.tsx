"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface ActiveRoomCardProps {
  roomId: string;
}

export function ActiveRoomCard({ roomId }: ActiveRoomCardProps) {
  const router = useRouter();

  return (
    <div className="max-w-md mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Create Room</CardTitle>
          <CardDescription>You already have an active room.</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              You are already in an active room. You can only be in one room at
              a time.
            </AlertDescription>
          </Alert>
        </CardContent>
        <CardFooter>
          <Button className="w-full" onClick={() => router.push(`/rooms/${roomId}`)}>
            Go to Room
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
