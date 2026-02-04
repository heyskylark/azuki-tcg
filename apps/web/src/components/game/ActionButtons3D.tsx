"use client";

import { useMemo, useState, useCallback } from "react";
import { Text } from "@react-three/drei";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { buildNoopAction, hasNoopAction } from "@/lib/game/actionValidation";

const ACTION_MULLIGAN = 23;

const BUTTON_Y = 0.3; // Slightly above hand cards (hand y = 0.05)
const BUTTON_Z = 8.1; // Slightly closer to player than the hand row
const BUTTON_THICKNESS = 0.08;
const BUTTON_HEIGHT = 0.6;
const BUTTON_GAP = 0.35;
const TEXT_SIZE = 0.22;

type ButtonVariant = "primary" | "secondary";

interface ActionButtonConfig {
  id: string;
  label: string;
  variant: ButtonVariant;
  onClick: () => void;
}

function getButtonWidth(label: string): number {
  return Math.max(2, label.length * 0.12 + 0.6);
}

function ActionButton3D({
  label,
  variant,
  onClick,
  position,
  width,
}: {
  label: string;
  variant: ButtonVariant;
  onClick: () => void;
  position: [number, number, number];
  width: number;
}) {
  const [hovered, setHovered] = useState(false);

  const baseColor = variant === "primary" ? "#3b82f6" : "#475569";
  const hoverColor = variant === "primary" ? "#60a5fa" : "#64748b";
  const textColor = "#f8fafc";

  return (
    <group
      position={position}
      onPointerOver={(event) => {
        event.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={(event) => {
        event.stopPropagation();
        setHovered(false);
      }}
      onPointerDown={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      <mesh castShadow receiveShadow>
        <boxGeometry args={[width, BUTTON_THICKNESS, BUTTON_HEIGHT]} />
        <meshStandardMaterial color={hovered ? hoverColor : baseColor} />
      </mesh>
      <Text
        position={[0, BUTTON_THICKNESS / 2 + 0.01, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={TEXT_SIZE}
        color={textColor}
        anchorX="center"
        anchorY="middle"
        maxWidth={width - 0.3}
        raycast={() => null}
      >
        {label}
      </Text>
    </group>
  );
}

export function ActionButtons3D() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  const actionMask = gameState?.actionMask ?? null;
  const legalPrimary = actionMask?.legalPrimary ?? [];

  const canNoop = hasNoopAction(actionMask);
  const canMulligan = legalPrimary.includes(ACTION_MULLIGAN);
  const isMulliganPhase = gameState?.phase === "PREGAME_MULLIGAN";

  const abilitySubphase = gameState?.abilitySubphase ?? "NONE";
  const canSkipTargetSelection =
    abilitySubphase === "EFFECT_SELECTION" && canNoop;

  const handleNoop = useCallback(() => {
    if (!canNoop) return;
    send({
      type: "GAME_ACTION",
      action: buildNoopAction(),
    });
  }, [canNoop, send]);

  const handleMulligan = useCallback(() => {
    if (!canMulligan) return;
    send({
      type: "GAME_ACTION",
      action: [ACTION_MULLIGAN, 0, 0, 0],
    });
  }, [canMulligan, send]);

  const buttons = useMemo<ActionButtonConfig[]>(() => {
    const nextButtons: ActionButtonConfig[] = [];

    if (canMulligan) {
      nextButtons.push({
        id: "mulligan",
        label: "Mulligan",
        variant: "primary",
        onClick: handleMulligan,
      });
    }

    if (canSkipTargetSelection) {
      nextButtons.push({
        id: "skip-target-selection",
        label: "Skip Target Selection",
        variant: "secondary",
        onClick: handleNoop,
      });
    } else if (canNoop && abilitySubphase === "NONE") {
      nextButtons.push({
        id: "noop",
        label: isMulliganPhase ? "Keep Hand" : "Pass",
        variant: "secondary",
        onClick: handleNoop,
      });
    }

    return nextButtons;
  }, [
    abilitySubphase,
    canMulligan,
    canNoop,
    canSkipTargetSelection,
    handleMulligan,
    handleNoop,
    isMulliganPhase,
  ]);

  if (buttons.length === 0) {
    return null;
  }

  const widths = buttons.map((button) => getButtonWidth(button.label));
  const totalWidth =
    widths.reduce((sum, width) => sum + width, 0) +
    BUTTON_GAP * (buttons.length - 1);

  let cursorX = -totalWidth / 2;

  return (
    <group position={[0, BUTTON_Y, BUTTON_Z]}>
      {buttons.map((button, index) => {
        const width = widths[index];
        const x = cursorX + width / 2;
        cursorX += width + BUTTON_GAP;

        return (
          <ActionButton3D
            key={button.id}
            label={button.label}
            variant={button.variant}
            onClick={button.onClick}
            position={[x, 0, 0]}
            width={width}
          />
        );
      })}
    </group>
  );
}
