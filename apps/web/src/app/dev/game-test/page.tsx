"use client";

import { useState, useEffect } from "react";
import { GameStateProvider, useGameState } from "@/contexts/GameStateContext";
import { AssetProvider } from "@/contexts/AssetContext";
import { GameScene } from "@/components/game/GameScene";
import {
  mockScenarios,
  getMockGameState,
  buildMockCardMappings,
  type MockScenario,
} from "@/app/dev/game-test/mockData";

/**
 * Scenario selector panel.
 */
function ScenarioSelector({
  scenario,
  onScenarioChange,
}: {
  scenario: MockScenario;
  onScenarioChange: (scenario: MockScenario) => void;
}) {
  return (
    <div className="absolute top-4 left-4 z-10 bg-black/80 p-4 rounded-lg border border-gray-700 max-w-xs">
      <h2 className="text-white text-lg font-bold mb-3">Test Scenarios</h2>
      <div className="space-y-2">
        {mockScenarios.map((s) => (
          <button
            key={s.id}
            onClick={() => onScenarioChange(s.id)}
            className={`w-full text-left px-3 py-2 rounded transition-colors ${
              scenario === s.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            <div className="font-medium">{s.name}</div>
            <div className="text-xs text-gray-400">{s.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * State info panel showing current game state.
 */
function StateInfoPanel() {
  const { gameState } = useGameState();

  if (!gameState) {
    return null;
  }

  return (
    <div className="absolute top-4 right-4 z-10 bg-black/80 p-4 rounded-lg border border-gray-700 max-w-xs">
      <h2 className="text-white text-lg font-bold mb-3">Game State</h2>
      <div className="text-sm text-gray-300 space-y-1">
        <div>
          <span className="text-gray-500">Phase:</span> {gameState.phase}
        </div>
        <div>
          <span className="text-gray-500">Turn:</span> {gameState.turnNumber}
        </div>
        <div>
          <span className="text-gray-500">Active:</span>{" "}
          {gameState.activePlayer === 0 ? "You" : "Opponent"}
        </div>
        <hr className="border-gray-700 my-2" />
        <div className="font-medium text-white">Your Board:</div>
        <div>
          <span className="text-gray-500">Leader HP:</span>{" "}
          {gameState.myBoard.leader.curHp}
        </div>
        <div>
          <span className="text-gray-500">Garden:</span>{" "}
          {gameState.myBoard.garden.filter(Boolean).length}/5
        </div>
        <div>
          <span className="text-gray-500">Hand:</span>{" "}
          {gameState.myHand.length} cards
        </div>
        <div>
          <span className="text-gray-500">IKZ:</span>{" "}
          {gameState.myBoard.ikzArea.filter((i) => !i.tapped).length}/
          {gameState.myBoard.ikzArea.length} untapped
        </div>
        <hr className="border-gray-700 my-2" />
        <div className="font-medium text-white">Opponent Board:</div>
        <div>
          <span className="text-gray-500">Leader HP:</span>{" "}
          {gameState.opponentBoard.leader.curHp}
        </div>
        <div>
          <span className="text-gray-500">Garden:</span>{" "}
          {gameState.opponentBoard.garden.filter(Boolean).length}/5
        </div>
        <div>
          <span className="text-gray-500">Hand:</span>{" "}
          {gameState.opponentBoard.handCount} cards
        </div>
      </div>
    </div>
  );
}

/**
 * Instructions panel.
 */
function InstructionsPanel() {
  return (
    <div className="absolute bottom-4 left-4 z-10 bg-black/80 p-3 rounded-lg border border-gray-700">
      <div className="text-sm text-gray-400">
        <span className="text-white font-medium">Controls:</span> Drag to
        rotate, Scroll to zoom
      </div>
    </div>
  );
}

/**
 * Inner component that has access to GameStateContext.
 */
function GameTestContent({
  scenario,
  onScenarioChange,
}: {
  scenario: MockScenario;
  onScenarioChange: (scenario: MockScenario) => void;
}) {
  const { setMockState, setCardMappings } = useGameState();

  // Load mock state when scenario changes
  useEffect(() => {
    const mockState = getMockGameState(scenario);
    const cardMappings = buildMockCardMappings();
    setCardMappings(cardMappings);
    setMockState(mockState);
  }, [scenario, setMockState, setCardMappings]);

  return (
    <>
      <GameScene />
      <ScenarioSelector scenario={scenario} onScenarioChange={onScenarioChange} />
      <StateInfoPanel />
      <InstructionsPanel />
    </>
  );
}

/**
 * Development test page for the 3D game renderer.
 * Allows testing with mock data without needing the WebSocket server.
 */
export default function GameTestPage() {
  const [scenario, setScenario] = useState<MockScenario>("early");

  return (
    <div className="h-screen w-screen bg-black">
      <AssetProvider>
        <GameStateProvider>
          <GameTestContent scenario={scenario} onScenarioChange={setScenario} />
        </GameStateProvider>
      </AssetProvider>
    </div>
  );
}
