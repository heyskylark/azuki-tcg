"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Board } from "@/components/game/board/Board";

/**
 * Main game scene component.
 * Renders the 3D canvas with the game board.
 */
export function GameScene() {
  return (
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
  );
}
