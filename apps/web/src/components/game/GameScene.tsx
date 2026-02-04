"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Board } from "@/components/game/board/Board";
import { DraggedCard } from "@/components/game/cards/DraggedCard";
import { AbilityOverlay } from "@/components/game/abilities/AbilityOverlay";
import { ActionButtons3D } from "@/components/game/ActionButtons3D";

/**
 * Main game scene component.
 * Renders the 3D canvas with the game board and ability UI overlay.
 */
export function GameScene() {
  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 28, 0.1], fov: 38 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        onCreated={(state) => {
          state.gl.setClearColor("#1a1a2a");
        }}
      >
        {/* Flat, bright lighting like Hearthstone */}
        <ambientLight intensity={1.8} />
        <directionalLight
          position={[0, 20, 0]}
          intensity={0.8}
        />
        {/* Soft fill lights from sides for even illumination */}
        <directionalLight position={[10, 10, 0]} intensity={0.3} />
        <directionalLight position={[-10, 10, 0]} intensity={0.3} />

        {/* Game content */}
        <Suspense fallback={null}>
          <Board />
          <DraggedCard />
          <ActionButtons3D />
        </Suspense>

        {/* Camera controls - limited to slight adjustments */}
        <OrbitControls
          enablePan={false}
          enableRotate={false}
          minDistance={15}
          maxDistance={30}
          target={[0, 0, 0]}
        />
      </Canvas>

      {/* Ability UI overlay - renders on top of 3D canvas */}
      <AbilityOverlay />
    </div>
  );
}
