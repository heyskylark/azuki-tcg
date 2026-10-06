import assert from "node:assert/strict";
import { test } from "node:test";
import { enqueueRoomMutation } from "@/state/RoomMutationQueue";

test("room mutations remain ordered across asynchronous persistence", async () => {
  const events: string[] = [];
  let releaseFirst: () => void = () => undefined;
  let markFirstStarted: () => void = () => undefined;
  const firstBarrier = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  const firstStarted = new Promise<void>((resolve) => {
    markFirstStarted = resolve;
  });

  const first = enqueueRoomMutation("room-ordered", async () => {
    events.push("first:start");
    markFirstStarted();
    await firstBarrier;
    events.push("first:end");
  });
  const second = enqueueRoomMutation("room-ordered", async () => {
    events.push("second");
  });

  await firstStarted;
  assert.deepEqual(events, ["first:start"]);
  releaseFirst();
  await Promise.all([first, second]);
  assert.deepEqual(events, ["first:start", "first:end", "second"]);
});

test("a rejected mutation does not strand the room queue", async () => {
  const failed = enqueueRoomMutation("room-rejected", async () => {
    throw new Error("expected failure");
  });
  const recovered = enqueueRoomMutation("room-rejected", async () => "continued");

  await assert.rejects(failed, /expected failure/);
  assert.equal(await recovered, "continued");
});
