"use client";

import { useRef, useState, type ReactNode } from "react";
import { useFrame, type ThreeElements } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import * as THREE from "three";
import { useAssets } from "@/contexts/AssetContext";
import { CardStats } from "@/components/game/cards/CardStats";

// Card dimensions in Three.js units
export const CARD_WIDTH = 1.4;
export const CARD_HEIGHT = 2.0;
export const CARD_DEPTH = 0.05;

interface Card3DProps {
  cardCode: string;
  imageUrl: string;
  name: string;
  attack?: number | null;
  health?: number | null;
  position: [number, number, number];
  tapped?: boolean;
  cooldown?: boolean;
  isFrozen?: boolean;
  isShocked?: boolean;
  onClick?: () => void;
  showStats?: boolean;
  children?: ReactNode;
}

/**
 * 3D card component with texture, stats display, and status effects.
 */
export function Card3D({
  cardCode,
  imageUrl,
  name,
  attack,
  health,
  position,
  tapped = false,
  cooldown = false,
  isFrozen = false,
  isShocked = false,
  onClick,
  showStats = true,
  children,
}: Card3DProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hovered, setHover] = useState(false);
  const { getCardTexture, cardBackTexture } = useAssets();

  // Get texture from cache or use placeholder
  const texture = getCardTexture(cardCode);

  // Animate tapped rotation (rotate around Y axis for flat cards on XZ plane)
  const targetRotationY = tapped ? -Math.PI / 2 : 0;

  useFrame((_state, delta) => {
    if (groupRef.current) {
      // Smoothly interpolate rotation
      const currentRotation = groupRef.current.rotation.y;
      const diff = targetRotationY - currentRotation;
      if (Math.abs(diff) > 0.01) {
        groupRef.current.rotation.y += diff * Math.min(delta * 8, 1);
      } else {
        groupRef.current.rotation.y = targetRotationY;
      }

      // Hover lift effect
      const targetY = hovered ? 0.2 : 0;
      const currentY = groupRef.current.position.y;
      const yDiff = targetY - currentY;
      if (Math.abs(yDiff) > 0.01) {
        groupRef.current.position.y += yDiff * Math.min(delta * 10, 1);
      }
    }
  });

  // Determine card color based on state
  const getCardColor = () => {
    if (isFrozen) return "#88ccff";
    if (isShocked) return "#ffff88";
    if (cooldown) return "#888888";
    if (hovered) return "#ffffff";
    return "#dddddd";
  };

  return (
    <group position={position}>
      {/* Animated group containing card + overlays + stats */}
      <group ref={groupRef}>
        <mesh
          onClick={(e) => {
            e.stopPropagation();
            onClick?.();
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHover(true);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={() => {
            setHover(false);
            document.body.style.cursor = "default";
          }}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[CARD_WIDTH, CARD_DEPTH, CARD_HEIGHT]} />
          {texture ? (
            <>
              {/* Side faces (edges) */}
              <meshStandardMaterial attach="material-0" color="#2a2a4e" />
              <meshStandardMaterial attach="material-1" color="#2a2a4e" />
              {/* Top face (card art) - faces up after X rotation */}
              <meshStandardMaterial
                attach="material-2"
                map={texture}
                color={getCardColor()}
              />
              {/* Bottom face (card back) - faces down after X rotation */}
              <meshStandardMaterial
                attach="material-3"
                map={cardBackTexture ?? undefined}
                color="#1a1a2e"
              />
              {/* Front/back edges */}
              <meshStandardMaterial attach="material-4" color="#2a2a4e" />
              <meshStandardMaterial attach="material-5" color="#2a2a4e" />
            </>
          ) : (
            // Placeholder material when texture not loaded
            <meshStandardMaterial color={getCardColor()} />
          )}
        </mesh>

        {/* Status effect overlays */}
        {isFrozen && <FrozenOverlay />}
        {isShocked && <ShockedOverlay />}
        {cooldown && <CooldownOverlay />}

        {/* Stats display */}
        {showStats && attack !== null && attack !== undefined && health !== null && health !== undefined && (
          <CardStats
            attack={attack}
            health={health}
            position={[0, CARD_DEPTH + 0.01, 0]}
            tapped={tapped}
          />
        )}

        {/* Additional elements passed as children */}
        {children}
      </group>
    </group>
  );
}

/**
 * Frozen effect overlay - blue tint.
 */
function FrozenOverlay() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, CARD_DEPTH + 0.01, 0]}>
      <planeGeometry args={[CARD_WIDTH * 0.9, CARD_HEIGHT * 0.9]} />
      <meshBasicMaterial
        color="#4488ff"
        transparent
        opacity={0.3}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Shocked effect overlay - yellow lightning.
 */
function ShockedOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulsing effect
      const pulse = Math.sin(state.clock.elapsedTime * 8) * 0.1 + 0.2;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.02, 0]}
    >
      <planeGeometry args={[CARD_WIDTH * 0.9, CARD_HEIGHT * 0.9]} />
      <meshBasicMaterial
        color="#ffff00"
        transparent
        opacity={0.2}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Cooldown overlay - grayscale effect indicator.
 */
function CooldownOverlay() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, CARD_DEPTH + 0.01, 0]}>
      <planeGeometry args={[CARD_WIDTH * 0.9, CARD_HEIGHT * 0.9]} />
      <meshBasicMaterial
        color="#000000"
        transparent
        opacity={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Placeholder card for empty slots.
 */
export function EmptyCardSlot({
  position,
  label,
}: {
  position: [number, number, number];
  label?: string;
}) {
  const [hovered, setHover] = useState(false);

  return (
    <group position={position}>
      <mesh
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <boxGeometry args={[CARD_WIDTH, CARD_DEPTH * 0.5, CARD_HEIGHT]} />
        <meshStandardMaterial
          color={hovered ? "#3a3a5e" : "#1a1a2e"}
          transparent
          opacity={0.5}
        />
      </mesh>
    </group>
  );
}
