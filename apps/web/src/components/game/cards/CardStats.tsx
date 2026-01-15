"use client";

import { Text } from "@react-three/drei";
import { CARD_WIDTH, CARD_HEIGHT } from "@/components/game/cards/Card3D";

interface CardStatsProps {
  attack: number;
  health: number;
  position?: [number, number, number];
  tapped?: boolean;
}

/**
 * WebGL text display for card attack and health values.
 * Positioned at the bottom of the card.
 */
export function CardStats({
  attack,
  health,
  position = [0, 0, 0],
  tapped = false,
}: CardStatsProps) {
  // Adjust position based on tapped state
  // When tapped, the card rotates -90 degrees on Y axis
  const statsOffset = CARD_HEIGHT * 0.35;

  return (
    <group position={position} rotation={tapped ? [0, -Math.PI / 2, 0] : [0, 0, 0]}>
      {/* Attack value (left/sword icon position) */}
      <group position={[-CARD_WIDTH * 0.3, 0.1, statsOffset]}>
        <StatBadge value={attack} color="#ff6b6b" />
      </group>

      {/* Health value (right/heart icon position) */}
      <group position={[CARD_WIDTH * 0.3, 0.1, statsOffset]}>
        <StatBadge value={health} color="#4ecdc4" />
      </group>
    </group>
  );
}

interface StatBadgeProps {
  value: number;
  color: string;
}

/**
 * Individual stat badge with background and text.
 */
function StatBadge({ value, color }: StatBadgeProps) {
  return (
    <group>
      {/* Background circle */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.18, 16]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.7} />
      </mesh>

      {/* Colored ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <ringGeometry args={[0.14, 0.18, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>

      {/* Text value */}
      <Text
        position={[0, 0.02, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {value}
      </Text>
    </group>
  );
}

/**
 * Leader health display - larger format for leader cards.
 */
export function LeaderHealthDisplay({
  currentHp,
  maxHp = 25,
  position = [0, 0, 0],
}: {
  currentHp: number;
  maxHp?: number;
  position?: [number, number, number];
}) {
  const healthPercent = currentHp / maxHp;
  const healthColor =
    healthPercent > 0.5 ? "#4ecdc4" : healthPercent > 0.25 ? "#ffd93d" : "#ff6b6b";

  return (
    <group position={position}>
      {/* Background */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.3, 24]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.8} />
      </mesh>

      {/* Health ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <ringGeometry args={[0.24, 0.3, 24]} />
        <meshBasicMaterial color={healthColor} />
      </mesh>

      {/* HP text */}
      <Text
        position={[0, 0.02, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="white"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {currentHp}
      </Text>
    </group>
  );
}
