"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useAssets } from "@/contexts/AssetContext";
import { CARD_WIDTH, CARD_HEIGHT, CARD_DEPTH } from "@/components/game/cards/Card3D";

// Must match Board.tsx layout constants
const SLOT_SPACING = 1.8;
const OPP_GARDEN_Z = -1.5;
const RIGHT_SIDE_X = 6;

const RETICLE_SIZE = CARD_WIDTH;
const RETICLE_Y = CARD_DEPTH + 0.08;
const LINE_Y = CARD_DEPTH + 0.06;
const ARROW_STRIP_WIDTH = CARD_WIDTH * 0.35;
const ARROW_SEGMENTS = 24;
const ARROW_REPEAT_UNIT = 0.6;
const BEND_MAX = 0.4;
const TARGET_HIT_MARGIN = 0.15;

interface AttackDragOverlayProps {
  isActive: boolean;
  attackerPosition: [number, number, number];
  initialPointerPosition?: [number, number, number];
  validTargets: Set<number>;
  onCommit: (targetIndex: number) => void;
  onCancel: () => void;
}

export function AttackDragOverlay({
  isActive,
  attackerPosition,
  initialPointerPosition,
  validTargets,
  onCommit,
  onCancel,
}: AttackDragOverlayProps) {
  const { camera, gl } = useThree();
  const { getUiTexture } = useAssets();

  const reticleRef = useRef<THREE.Mesh>(null!);
  const arrowGroupRef = useRef<THREE.Group>(null!);
  const arrowMeshRef = useRef<THREE.Mesh>(null!);
  const reticleMaterialRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const arrowMaterialRef = useRef<THREE.MeshBasicMaterial | null>(null);

  const pointerPos = useRef(new THREE.Vector3());
  const hoveredTarget = useRef<number | null>(null);
  const isOverValidTarget = useRef(false);
  const lastLength = useRef(0);

  const tempStart = useRef(new THREE.Vector3());
  const tempEnd = useRef(new THREE.Vector3());
  const tempDir = useRef(new THREE.Vector3());

  const reticleTexture = getUiTexture("target");
  const arrowTexture = getUiTexture("target-pointer");

  useEffect(() => {
    if (arrowTexture) {
      arrowTexture.wrapS = THREE.RepeatWrapping;
      arrowTexture.wrapT = THREE.RepeatWrapping;
      arrowTexture.needsUpdate = true;
    }
  }, [arrowTexture]);

  const arrowGeometry = useMemo(() => {
    const geometry = new THREE.PlaneGeometry(1, ARROW_STRIP_WIDTH, ARROW_SEGMENTS, 1);
    // Shift so the strip starts at x=0 and ends at x=1 (anchor at start)
    geometry.translate(0.5, 0, 0);
    const positions = geometry.attributes.position as THREE.BufferAttribute;
    const basePositions = new Float32Array(positions.array);
    return { geometry, basePositions };
  }, []);

  useEffect(() => {
    return () => {
      arrowGeometry.geometry.dispose();
    };
  }, [arrowGeometry]);

  const lightRed = useMemo(() => new THREE.Color("#ff9a9a"), []);
  const deepRed = useMemo(() => new THREE.Color("#b30000"), []);

  const projectToXZPlane = useCallback(
    (clientX: number, clientY: number, targetY: number): THREE.Vector3 => {
      const rect = gl.domElement.getBoundingClientRect();
      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetY);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(plane, intersection);

      return intersection;
    },
    [camera, gl]
  );

  const getTargetIndexAtPosition = useCallback(
    (pos: THREE.Vector3): number | null => {
      const halfWidth = CARD_WIDTH / 2 + TARGET_HIT_MARGIN;
      const halfHeight = CARD_HEIGHT / 2 + TARGET_HIT_MARGIN;

      if (validTargets.has(5)) {
        const dx = Math.abs(pos.x - RIGHT_SIDE_X);
        const dz = Math.abs(pos.z - OPP_GARDEN_Z);
        if (dx <= halfWidth && dz <= halfHeight) {
          return 5;
        }
      }

      for (let i = 0; i < 5; i++) {
        if (!validTargets.has(i)) continue;
        const targetX = (i - 2) * SLOT_SPACING;
        const dx = Math.abs(pos.x - targetX);
        const dz = Math.abs(pos.z - OPP_GARDEN_Z);
        if (dx <= halfWidth && dz <= halfHeight) {
          return i;
        }
      }

      return null;
    },
    [validTargets]
  );

  useEffect(() => {
    if (!isActive) return;

    const initialX = initialPointerPosition?.[0] ?? attackerPosition[0];
    const initialZ = initialPointerPosition?.[2] ?? attackerPosition[2];
    pointerPos.current.set(initialX, RETICLE_Y, initialZ);

    const initialTarget = getTargetIndexAtPosition(pointerPos.current);
    hoveredTarget.current = initialTarget;
    isOverValidTarget.current = initialTarget !== null;
  }, [attackerPosition, getTargetIndexAtPosition, initialPointerPosition, isActive]);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      if (!isActive) return;

      const intersection = projectToXZPlane(
        event.clientX,
        event.clientY,
        RETICLE_Y
      );
      pointerPos.current.copy(intersection);

      const targetIndex = getTargetIndexAtPosition(intersection);
      hoveredTarget.current = targetIndex;
      isOverValidTarget.current = targetIndex !== null;
    },
    [getTargetIndexAtPosition, isActive, projectToXZPlane]
  );

  const handlePointerUp = useCallback(() => {
    if (!isActive) return;

    const targetIndex = hoveredTarget.current;
    if (targetIndex !== null) {
      onCommit(targetIndex);
    } else {
      onCancel();
    }
  }, [isActive, onCancel, onCommit]);

  useEffect(() => {
    if (!isActive) return;

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [handlePointerMove, handlePointerUp, isActive]);

  useFrame(() => {
    if (!isActive) return;

    const reticle = reticleRef.current;
    if (reticle) {
      reticle.position.set(pointerPos.current.x, RETICLE_Y, pointerPos.current.z);
    }

    const start = tempStart.current.set(attackerPosition[0], LINE_Y, attackerPosition[2]);
    const end = tempEnd.current.set(pointerPos.current.x, LINE_Y, pointerPos.current.z);
    const dir = tempDir.current.subVectors(end, start);
    const length = dir.length();

    const arrowGroup = arrowGroupRef.current;
    const arrowMesh = arrowMeshRef.current;
    if (arrowGroup && arrowMesh) {
      if (length < 0.1) {
        arrowGroup.visible = false;
      } else {
        arrowGroup.visible = true;
        const angle = -Math.atan2(dir.z, dir.x);
        arrowGroup.position.set(start.x, LINE_Y, start.z);
        arrowGroup.rotation.set(0, angle, 0);
        arrowMesh.scale.set(length, 1, 1);
      }
    }

    if (arrowTexture && Math.abs(length - lastLength.current) > 0.02) {
      const repeats = Math.max(1, length / ARROW_REPEAT_UNIT);
      arrowTexture.repeat.set(repeats, 1);
      arrowTexture.needsUpdate = true;
      lastLength.current = length;
    }

    if (arrowMesh) {
      const positions = arrowGeometry.geometry.attributes.position as THREE.BufferAttribute;
      const basePositions = arrowGeometry.basePositions;
      const bendFactor = THREE.MathUtils.clamp(pointerPos.current.x / 6, -1, 1);
      const bendAmount = bendFactor * BEND_MAX;

      for (let i = 0; i < positions.count; i++) {
        const baseX = basePositions[i * 3];
        const baseY = basePositions[i * 3 + 1];
        const baseZ = basePositions[i * 3 + 2];
        const t = baseX;
        const bend = Math.sin(t * Math.PI) * bendAmount;
        positions.setXYZ(i, baseX, baseY + bend, baseZ);
      }
      positions.needsUpdate = true;
    }

    const targetColor = isOverValidTarget.current ? deepRed : lightRed;
    const targetOpacity = isOverValidTarget.current ? 0.85 : 0.55;

    if (reticleMaterialRef.current) {
      reticleMaterialRef.current.color.lerp(targetColor, 0.15);
      reticleMaterialRef.current.opacity +=
        (targetOpacity - reticleMaterialRef.current.opacity) * 0.15;
    }

    if (arrowMaterialRef.current) {
      arrowMaterialRef.current.color.lerp(targetColor, 0.15);
      arrowMaterialRef.current.opacity +=
        (targetOpacity - arrowMaterialRef.current.opacity) * 0.15;
    }
  });

  if (!isActive) {
    return null;
  }

  return (
    <group>
      {/* Arrow strip */}
      <group ref={arrowGroupRef}>
        <mesh ref={arrowMeshRef} rotation={[-Math.PI / 2, 0, 0]} renderOrder={10} raycast={() => {}}>
          <primitive object={arrowGeometry.geometry} attach="geometry" />
          <meshBasicMaterial
            ref={arrowMaterialRef}
            color="#ff9a9a"
            transparent
            opacity={0.55}
            alphaMap={arrowTexture ?? undefined}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      </group>

      {/* Target reticle */}
      <mesh ref={reticleRef} rotation={[-Math.PI / 2, 0, 0]} renderOrder={11} raycast={() => {}}>
        <planeGeometry args={[RETICLE_SIZE, RETICLE_SIZE]} />
        <meshBasicMaterial
          ref={reticleMaterialRef}
          color="#ff9a9a"
          transparent
          opacity={0.55}
          alphaMap={reticleTexture ?? undefined}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}
