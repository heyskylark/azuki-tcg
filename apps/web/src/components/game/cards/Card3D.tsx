"use client";

import { useRef, useState, type ReactNode } from "react";
import { Text } from "@react-three/drei";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useAssets } from "@/contexts/AssetContext";
import { CardStats } from "@/components/game/cards/CardStats";
import { useDragStore } from "@/stores/dragStore";

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
  isEffectImmune?: boolean;
  hasCharge?: boolean;
  hasDefender?: boolean;
  hasInfiltrate?: boolean;
  onClick?: () => void;
  showStats?: boolean;
  children?: ReactNode;
  // Ability targeting props
  isAbilityTarget?: boolean;
  onAbilityTargetClick?: () => void;
  // Ability activation props
  isAbilityActivatable?: boolean;
  onAbilityActivate?: () => void;
  // Weapon attachment targeting props
  isWeaponTarget?: boolean;
  onWeaponTargetClick?: () => void;
  // Attack targeting props
  isAttackTarget?: boolean;
  // Defender targeting props
  isDefenderTarget?: boolean;
  onDefenderTargetClick?: () => void;
  onPointerDown?: (event: ThreeEvent<PointerEvent>) => void;
  onPointerUp?: (event: ThreeEvent<PointerEvent>) => void;
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
  isEffectImmune = false,
  hasCharge = false,
  hasDefender = false,
  hasInfiltrate = false,
  onClick,
  showStats = true,
  children,
  isAbilityTarget = false,
  onAbilityTargetClick,
  isAbilityActivatable = false,
  onAbilityActivate,
  isWeaponTarget = false,
  onWeaponTargetClick,
  isAttackTarget = false,
  isDefenderTarget = false,
  onDefenderTargetClick,
  onPointerDown,
  onPointerUp,
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
    if (isEffectImmune) return "#88ffdd";
    if (cooldown) return "#888888";
    if (hovered) return "#ffffff";
    return "#dddddd";
  };

  const keywordBadges = [
    hasCharge ? { label: "CHG", color: "#ffd166" } : null,
    hasDefender ? { label: "DEF", color: "#4ecdc4" } : null,
    hasInfiltrate ? { label: "INF", color: "#ff6b6b" } : null,
  ].filter(Boolean) as Array<{ label: string; color: string }>;

  return (
    <group position={position}>
      {/* Animated group containing card + overlays + stats */}
      <group ref={groupRef}>
        <mesh
          onClick={(e) => {
            e.stopPropagation();
            // If this is a weapon target and we have a target click handler, use it
            if (isWeaponTarget && onWeaponTargetClick) {
              onWeaponTargetClick();
            } else if (isAbilityTarget && onAbilityTargetClick) {
              // If this is an ability target and we have a target click handler, use it
              onAbilityTargetClick();
            } else if (isDefenderTarget && onDefenderTargetClick) {
              onDefenderTargetClick();
            } else if (isAbilityActivatable && onAbilityActivate) {
              onAbilityActivate();
            } else {
              onClick?.();
            }
          }}
          onPointerDown={(e) => {
            onPointerDown?.(e);
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHover(true);
            document.body.style.cursor =
              isWeaponTarget ||
              isAbilityTarget ||
              isAttackTarget ||
              isAbilityActivatable ||
              isDefenderTarget
                ? "crosshair"
                : "pointer";
          }}
          onPointerOut={() => {
            setHover(false);
            document.body.style.cursor = "default";
          }}
          onPointerUp={(e) => {
            // Handle weapon attachment via drop (pointer up while hovering)
            if (isWeaponTarget && onWeaponTargetClick) {
              e.stopPropagation();
              onWeaponTargetClick();
            }
            onPointerUp?.(e);
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
        {isEffectImmune && <EffectImmuneOverlay />}
        {cooldown && <CooldownOverlay />}

        {/* Keyword badges */}
        {keywordBadges.length > 0 && <KeywordBadges badges={keywordBadges} />}

        {/* Ability activation badge */}
        {isAbilityActivatable && onAbilityActivate && (
          <AbilityActivateBadge onActivate={onAbilityActivate} />
        )}

        {/* Ability activation highlight */}
        {isAbilityActivatable && <AbilityActivateOverlay />}

        {/* Ability target highlight */}
        {isAbilityTarget && <AbilityTargetOverlay />}

        {/* Weapon target highlight */}
        {isWeaponTarget && <WeaponTargetOverlay />}

        {/* Attack target highlight */}
        {isAttackTarget && <AttackTargetOverlay />}

        {/* Defender target highlight */}
        {isDefenderTarget && <DefenderTargetOverlay />}

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

function KeywordBadges({
  badges,
}: {
  badges: Array<{ label: string; color: string }>;
}) {
  const startX = -CARD_WIDTH * 0.38;
  const startZ = CARD_HEIGHT * 0.38;
  const gap = 0.34;

  return (
    <group>
      {badges.map((badge, index) => (
        <group
          key={badge.label}
          position={[startX + index * gap, CARD_DEPTH + 0.012, startZ]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <mesh>
            <planeGeometry args={[0.32, 0.16]} />
            <meshBasicMaterial color="#0b0b0b" transparent opacity={0.75} />
          </mesh>
          <Text
            position={[0, 0.001, 0]}
            fontSize={0.12}
            color={badge.color}
            anchorX="center"
            anchorY="middle"
          >
            {badge.label}
          </Text>
        </group>
      ))}
    </group>
  );
}

function AbilityActivateBadge({ onActivate }: { onActivate: () => void }) {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 6) * 0.12 + 0.7;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <group
      position={[CARD_WIDTH * 0.36, CARD_DEPTH + 0.05, CARD_HEIGHT * 0.36]}
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <mesh
        ref={meshRef}
        onPointerDown={(e) => {
          e.stopPropagation();
        }}
        onClick={(e) => {
          e.stopPropagation();
          onActivate();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          document.body.style.cursor = "default";
        }}
      >
        <circleGeometry args={[0.18, 24]} />
        <meshBasicMaterial color="#33ccff" transparent opacity={0.75} />
      </mesh>
      <Text
        position={[0, 0.001, 0]}
        fontSize={0.16}
        color="#061622"
        anchorX="center"
        anchorY="middle"
        raycast={() => null}
      >
        A
      </Text>
    </group>
  );
}

/**
 * Ability activation overlay - pulsing cyan highlight for activatable abilities.
 */
function AbilityActivateOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 5) * 0.12 + 0.25;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.02, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH + 0.24, CARD_HEIGHT + 0.24]} />
      <meshBasicMaterial
        color="#33ccff"
        transparent
        opacity={0.25}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Frozen effect overlay - blue tint.
 */
function FrozenOverlay() {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.01, 0]}
      raycast={() => null}
    >
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
      raycast={() => null}
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
 * EffectImmune overlay - teal shield tint.
 */
function EffectImmuneOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 6) * 0.1 + 0.25;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.015, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH * 0.92, CARD_HEIGHT * 0.92]} />
      <meshBasicMaterial
        color="#33ddb9"
        transparent
        opacity={0.25}
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
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.01, 0]}
      raycast={() => null}
    >
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
 * Ability target overlay - pulsing golden highlight for valid targets.
 */
function AbilityTargetOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulsing effect for ability targets
      const pulse = Math.sin(state.clock.elapsedTime * 4) * 0.15 + 0.35;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.03, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH + 0.2, CARD_HEIGHT + 0.2]} />
      <meshBasicMaterial
        color="#ffaa00"
        transparent
        opacity={0.35}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Weapon target overlay - pulsing orange highlight for valid weapon attachment targets.
 */
function WeaponTargetOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulsing effect for weapon targets
      const pulse = Math.sin(state.clock.elapsedTime * 5) * 0.15 + 0.4;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.03, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH + 0.2, CARD_HEIGHT + 0.2]} />
      <meshBasicMaterial
        color="#ff6600"
        transparent
        opacity={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Attack target overlay - pulsing red highlight for valid attack targets.
 */
function AttackTargetOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 5) * 0.15 + 0.45;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.03, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH + 0.2, CARD_HEIGHT + 0.2]} />
      <meshBasicMaterial
        color="#cc3333"
        transparent
        opacity={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Defender target overlay - pulsing teal highlight for valid defenders.
 */
function DefenderTargetOverlay() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 5) * 0.15 + 0.45;
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
    }
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, CARD_DEPTH + 0.03, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[CARD_WIDTH + 0.2, CARD_HEIGHT + 0.2]} />
      <meshBasicMaterial
        color="#4ecdc4"
        transparent
        opacity={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

interface EmptyCardSlotProps {
  position: [number, number, number];
  label?: string;
  slotIndex?: number;
  zone?: "garden" | "alley";
  isValidDropTarget?: boolean;
  onDrop?: () => void;
}

/**
 * Placeholder card for empty slots.
 * Highlights green when it's a valid drop target during drag.
 */
export function EmptyCardSlot({
  position,
  label,
  slotIndex,
  zone,
  isValidDropTarget = false,
  onDrop,
}: EmptyCardSlotProps) {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null!);

  const dragPhase = useDragStore((state) => state.dragPhase);
  const dragSourceType = useDragStore((state) => state.dragSourceType);
  const setHoveredSlot = useDragStore((state) => state.setHoveredSlot);

  const isDragging = dragPhase === "dragging" || dragPhase === "pickup";
  const showValidHighlight = isDragging && isValidDropTarget;
  const isHoveredValidTarget = showValidHighlight && hovered;

  // Animate glow effect for valid drop targets
  useFrame((state) => {
    if (meshRef.current && showValidHighlight) {
      const pulse = Math.sin(state.clock.elapsedTime * 4) * 0.15 + 0.85;
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.emissiveIntensity = isHoveredValidTarget ? 0.8 : pulse * 0.4;
    }
  });

  // Determine color based on state
  const getColor = () => {
    if (isHoveredValidTarget) return "#44ff44"; // Bright green when hovering valid target
    if (showValidHighlight) return "#228822"; // Green pulse for valid targets
    if (hovered) return "#3a3a5e";
    return "#1a1a2e";
  };

  const handlePointerOver = () => {
    setHovered(true);
    if (
      isDragging &&
      dragSourceType !== "spell" &&
      zone !== undefined &&
      slotIndex !== undefined
    ) {
      setHoveredSlot(zone, slotIndex);
    }
  };

  const handlePointerOut = () => {
    setHovered(false);
    if (isDragging && dragSourceType !== "spell") {
      setHoveredSlot(null, null);
    }
  };

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (isDragging && isValidDropTarget && onDrop) {
      onDrop();
    }
  };

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onPointerUp={handlePointerUp}
      >
        <boxGeometry args={[CARD_WIDTH, CARD_DEPTH * 0.5, CARD_HEIGHT]} />
        <meshStandardMaterial
          color={getColor()}
          transparent
          opacity={showValidHighlight ? 0.8 : 0.5}
          emissive={showValidHighlight ? "#22aa22" : "#000000"}
          emissiveIntensity={0}
        />
      </mesh>
    </group>
  );
}
