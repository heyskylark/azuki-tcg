const roomMutationTails = new Map<string, Promise<void>>();

export function enqueueRoomMutation<T>(roomId: string, mutation: () => Promise<T>): Promise<T> {
  const previous = roomMutationTails.get(roomId) ?? Promise.resolve();
  const result = previous.catch(() => undefined).then(mutation);
  const tail = result.then(
    () => undefined,
    () => undefined
  );
  roomMutationTails.set(roomId, tail);
  void tail.finally(() => {
    if (roomMutationTails.get(roomId) === tail) {
      roomMutationTails.delete(roomId);
    }
  });
  return result;
}
