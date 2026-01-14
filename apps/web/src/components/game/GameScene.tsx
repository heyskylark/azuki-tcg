"use client";

import { Suspense, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeElements } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { useGameState } from "@/contexts/GameStateContext";

/**
 * Test cube component for verifying the scene works.
 */
function TestCube(props: ThreeElements["mesh"]) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const [hovered, setHover] = useState(false);
  const [active, setActive] = useState(false);

  useFrame((_state, delta) => {
    meshRef.current.rotation.x += delta * 0.5;
    meshRef.current.rotation.y += delta * 0.3;
  });

  return (
    <mesh
      {...props}
      ref={meshRef}
      scale={active ? 1.5 : 1}
      onClick={() => setActive(!active)}
      onPointerOver={() => setHover(true)}
      onPointerOut={() => setHover(false)}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={hovered ? "hotpink" : "orange"} />
    </mesh>
  );
}

/**
 * Board surface placeholder.
 */
function BoardSurface() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]} receiveShadow>
      <planeGeometry args={[16, 12]} />
      <meshStandardMaterial color="#1a472a" />
    </mesh>
  );
}

/**
 * Placeholder card for testing.
 */
function PlaceholderCard({
  position,
  label,
  tapped = false,
}: {
  position: [number, number, number];
  label: string;
  tapped?: boolean;
}) {
  const [hovered, setHover] = useState(false);

  // Tapped cards rotate 90 degrees
  const rotation: [number, number, number] = tapped
    ? [-Math.PI / 2, Math.PI / 2, 0]
    : [-Math.PI / 2, 0, 0];

  return (
    <group position={position}>
      <mesh
        rotation={rotation}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
        castShadow
      >
        <boxGeometry args={[1.4, 0.05, 2.0]} />
        <meshStandardMaterial color={hovered ? "#4a9eff" : "#2a2a4e"} />
      </mesh>
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  );
}

/**
 * Game board with placeholder cards showing the layout.
 */
function GameBoard() {
  const { gameState } = useGameState();

  // Card slot positions
  const GARDEN_Y = 0;
  const GARDEN_Z_MY = 2; // My garden (bottom)
  const GARDEN_Z_OPP = -2; // Opponent garden (top)
  const SLOT_SPACING = 2;
  const LEADER_X = -6;
  const GATE_X = 6;

  return (
    <group>
      {/* Board surface */}
      <BoardSurface />

      {/* My side - Leader and Gate */}
      <PlaceholderCard position={[LEADER_X, GARDEN_Y, 4]} label="Leader" />
      <PlaceholderCard position={[GATE_X, GARDEN_Y, 4]} label="Gate" />

      {/* My Garden (5 slots) */}
      {[0, 1, 2, 3, 4].map((i) => (
        <PlaceholderCard
          key={`my-garden-${i}`}
          position={[(i - 2) * SLOT_SPACING, GARDEN_Y, GARDEN_Z_MY]}
          label={`G${i}`}
          tapped={i === 2} // Test tapped state
        />
      ))}

      {/* Opponent side - Leader and Gate */}
      <PlaceholderCard position={[LEADER_X, GARDEN_Y, -4]} label="Opp Leader" />
      <PlaceholderCard position={[GATE_X, GARDEN_Y, -4]} label="Opp Gate" />

      {/* Opponent Garden (5 slots) */}
      {[0, 1, 2, 3, 4].map((i) => (
        <PlaceholderCard
          key={`opp-garden-${i}`}
          position={[(i - 2) * SLOT_SPACING, GARDEN_Y, GARDEN_Z_OPP]}
          label={`O${i}`}
        />
      ))}

      {/* Phase indicator */}
      <Text
        position={[0, 3, -5]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="black"
      >
        {gameState ? `Phase: ${gameState.phase} | Turn ${gameState.turnNumber}` : "No Game State"}
      </Text>

      {/* Test cube in center */}
      <TestCube position={[0, 1, 0]} />
    </group>
  );
}

/**
 * Main game scene component.
 * Renders the 3D canvas with the game board.
 */
export function GameScene() {
  return (
    <Canvas
      camera={{ position: [0, 10, 12], fov: 50 }}
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true }}
      onCreated={(state) => {
        state.gl.setClearColor("#0a0a0a");
      }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[5, 10, 5]}
        intensity={1}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <pointLight position={[-5, 5, -5]} intensity={0.5} />

      {/* Game content */}
      <Suspense fallback={null}>
        <GameBoard />
      </Suspense>

      {/* Camera controls */}
      <OrbitControls
        enablePan={false}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI / 2.2}
        minDistance={8}
        maxDistance={20}
      />
    </Canvas>
  );
}
