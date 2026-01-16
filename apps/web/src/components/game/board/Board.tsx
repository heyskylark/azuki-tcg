"use client";

import { Text } from "@react-three/drei";
import { useGameState } from "@/contexts/GameStateContext";
import { Card3D, EmptyCardSlot, CARD_WIDTH, CARD_HEIGHT } from "@/components/game/cards/Card3D";
import { LeaderHealthDisplay } from "@/components/game/cards/CardStats";
import type {
  ResolvedPlayerBoard,
  ResolvedCard,
  ResolvedLeader,
  ResolvedGate,
  ResolvedHandCard,
  ResolvedIkz,
} from "@/types/game";

// Board layout constants
const SLOT_SPACING = 1.8;
const IKZ_SPACING = 1.2;  // Tighter spacing for IKZ cards (up to 10)
const GARDEN_SLOTS = 5;
const ALLEY_SLOTS = 5;

// Z positions (depth into screen) - positive Z is toward player
// Card height is 2.0, gap of ~2.1 between rows
const MY_GARDEN_Z = 1.5;
const MY_ALLEY_Z = 3.6;   // Same gap as alley-to-IKZ
const MY_IKZ_Z = 5.7;     // Same gap as garden-to-alley
const MY_HAND_Z = 7.4;    // Hand slightly overlaps IKZ area

const OPP_GARDEN_Z = -1.5;
const OPP_ALLEY_Z = -3.6;  // Same gap as alley-to-IKZ
const OPP_IKZ_Z = -5.7;    // Same gap as garden-to-alley

// X positions - right side for leader/gate/deck/discard, left side for IKZ pile
const RIGHT_SIDE_X = 6;
const DECK_X = RIGHT_SIDE_X;
const DISCARD_X = RIGHT_SIDE_X + 1.8;
const IKZ_PILE_X = -6;

/**
 * Board surface - the green felt table.
 */
function BoardSurface() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
      <planeGeometry args={[22, 20]} />
      <meshStandardMaterial color="#1a472a" />
    </mesh>
  );
}

/**
 * Render a row of garden/alley slots with cards or empty slots.
 */
function CardRow({
  cards,
  basePosition,
  isOpponent = false,
}: {
  cards: (ResolvedCard | null)[];
  basePosition: [number, number, number];
  isOpponent?: boolean;
}) {
  const [baseX, baseY, baseZ] = basePosition;

  return (
    <group>
      {cards.map((card, index) => {
        const x = (index - Math.floor(GARDEN_SLOTS / 2)) * SLOT_SPACING + baseX;
        const position: [number, number, number] = [x, baseY, baseZ];

        if (card) {
          return (
            <Card3D
              key={`card-${index}-${card.cardCode}`}
              cardCode={card.cardCode}
              imageUrl={card.imageUrl}
              name={card.name}
              attack={card.curAtk}
              health={card.curHp}
              position={position}
              tapped={card.tapped}
              cooldown={card.cooldown}
              isFrozen={card.isFrozen}
              isShocked={card.isShocked}
              showStats={true}
            />
          );
        }

        return (
          <EmptyCardSlot
            key={`empty-${index}`}
            position={position}
            label={`${isOpponent ? "O" : "G"}${index}`}
          />
        );
      })}
    </group>
  );
}

/**
 * Leader card with health display.
 */
function LeaderCard({
  leader,
  position,
}: {
  leader: ResolvedLeader;
  position: [number, number, number];
}) {
  return (
    <Card3D
      cardCode={leader.cardCode}
      imageUrl={leader.imageUrl}
      name={leader.name}
      attack={leader.curAtk}
      health={leader.curHp}
      position={position}
      tapped={leader.tapped}
      cooldown={leader.cooldown}
      showStats={false}
    >
      {/* Large health display above leader */}
      <LeaderHealthDisplay
        currentHp={leader.curHp}
        position={[0, 0.2, -1.2]}
      />
    </Card3D>
  );
}

/**
 * Gate card display.
 */
function GateCard({
  gate,
  position,
}: {
  gate: ResolvedGate;
  position: [number, number, number];
}) {
  return (
    <Card3D
      cardCode={gate.cardCode}
      imageUrl={gate.imageUrl}
      name={gate.name}
      position={position}
      tapped={gate.tapped}
      cooldown={gate.cooldown}
      showStats={false}
    />
  );
}

/**
 * IKZ pool display - IKZ cards spread in a row.
 */
function IkzPool({
  ikzArea,
  position,
}: {
  ikzArea: ResolvedIkz[];
  position: [number, number, number];
}) {
  const cardCount = ikzArea.length;

  return (
    <group position={position}>
      {ikzArea.map((ikz, index) => {
        // Center the row of cards with tighter spacing
        const x = (index - (cardCount - 1) / 2) * IKZ_SPACING;
        // Slight height increase per card to prevent z-fighting when overlapping
        const y = index * 0.01;
        return (
          <Card3D
            key={`ikz-${index}-${ikz.cardCode}`}
            cardCode={ikz.cardCode}
            imageUrl={ikz.imageUrl}
            name={ikz.name}
            position={[x, y, 0]}
            tapped={ikz.tapped}
            cooldown={ikz.cooldown}
            showStats={false}
          />
        );
      })}
    </group>
  );
}

/**
 * Deck stack representation.
 */
function DeckStack({
  count,
  position,
  label,
}: {
  count: number;
  position: [number, number, number];
  label: string;
}) {
  const stackHeight = 0.1 + count * 0.005;
  return (
    <group position={position}>
      {/* Stack of cards */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#2a2a4e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="white"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * Discard pile representation.
 */
function DiscardPile({
  count,
  position,
}: {
  count: number;
  position: [number, number, number];
}) {
  const stackHeight = Math.max(0.02, count * 0.003);
  return (
    <group position={position}>
      {/* Pile of cards - slightly messier than deck */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#3a2a2e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="#aa8888"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * IKZ pile stack representation.
 */
function IkzPileStack({
  count,
  position,
}: {
  count: number;
  position: [number, number, number];
}) {
  const stackHeight = 0.1 + count * 0.005;
  return (
    <group position={position}>
      {/* Stack of IKZ cards */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#2a4a2e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="#88ff88"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * Player's hand display - fan layout at bottom of screen.
 */
function HandDisplay({
  cards,
  position,
}: {
  cards: ResolvedHandCard[];
  position: [number, number, number];
}) {
  const cardCount = cards.length;

  return (
    <group position={position}>
      {cards.map((card, index) => {
        // Fan layout calculation
        const centerOffset = (cardCount - 1) / 2;
        const normalizedIndex = index - centerOffset;
        const x = normalizedIndex * (SLOT_SPACING * 0.7);
        // Slight height increase per card to prevent z-fighting when overlapping
        const y = index * 0.01;
        const z = Math.abs(normalizedIndex) * 0.1; // Slight arc
        const rotationY = normalizedIndex * -0.05; // Fan angle

        return (
          <group
            key={`hand-${index}-${card.cardCode}`}
            position={[x, y, z]}
            rotation={[0, rotationY, 0]}
          >
            <Card3D
              cardCode={card.cardCode}
              imageUrl={card.imageUrl}
              name={card.name}
              position={[0, 0, 0]}
              showStats={false}
            >
              {/* IKZ cost badge */}
              <group position={[CARD_WIDTH * 0.35, 0.1, -0.8]}>
                <mesh rotation={[-Math.PI / 2, 0, 0]}>
                  <circleGeometry args={[0.15, 16]} />
                  <meshBasicMaterial color="#4a4a8e" />
                </mesh>
                <Text
                  position={[0, 0.02, 0]}
                  rotation={[-Math.PI / 2, 0, 0]}
                  fontSize={0.15}
                  color="#88ff88"
                  anchorX="center"
                  anchorY="middle"
                  fontWeight="bold"
                >
                  {card.ikzCost}
                </Text>
              </group>
            </Card3D>
          </group>
        );
      })}
    </group>
  );
}

/**
 * One player's side of the board.
 */
function PlayerArea({
  board,
  hand,
  isOpponent,
}: {
  board: ResolvedPlayerBoard;
  hand?: ResolvedHandCard[];
  isOpponent: boolean;
}) {
  const gardenZ = isOpponent ? OPP_GARDEN_Z : MY_GARDEN_Z;
  const alleyZ = isOpponent ? OPP_ALLEY_Z : MY_ALLEY_Z;
  const ikzZ = isOpponent ? OPP_IKZ_Z : MY_IKZ_Z;

  return (
    <group>
      {/* Leader - right side, same row as garden */}
      <LeaderCard leader={board.leader} position={[RIGHT_SIDE_X, 0, gardenZ]} />

      {/* Gate - right side, same row as alley */}
      <GateCard gate={board.gate} position={[RIGHT_SIDE_X, 0, alleyZ]} />

      {/* Garden (front row) */}
      <CardRow cards={board.garden} basePosition={[0, 0, gardenZ]} isOpponent={isOpponent} />

      {/* Alley (back row) */}
      <CardRow cards={board.alley} basePosition={[0, 0, alleyZ]} isOpponent={isOpponent} />

      {/* Deck - below the gate */}
      <DeckStack
        count={board.deckCount}
        position={[DECK_X, 0, ikzZ]}
        label={isOpponent ? "Opp Deck" : "My Deck"}
      />

      {/* Discard pile - to the right of deck */}
      <DiscardPile
        count={board.discardCount}
        position={[DISCARD_X, 0, ikzZ]}
      />

      {/* IKZ Pile - to the left of IKZ area */}
      <IkzPileStack
        count={board.ikzPileCount}
        position={[IKZ_PILE_X, 0, ikzZ]}
      />

      {/* IKZ Pool - spread out below alley */}
      <IkzPool
        ikzArea={board.ikzArea}
        position={[0, 0, ikzZ]}
      />

      {/* Hand (only for player, not opponent) - slightly elevated to overlap IKZ */}
      {!isOpponent && hand && hand.length > 0 && (
        <HandDisplay cards={hand} position={[0, 0.05, MY_HAND_Z]} />
      )}
    </group>
  );
}

/**
 * Main game board component.
 * Renders the complete game state with both players' boards.
 */
export function Board() {
  const { gameState } = useGameState();

  if (!gameState) {
    return (
      <group>
        <BoardSurface />
        <Text
          position={[0, 1, 0]}
          fontSize={0.5}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          Waiting for game state...
        </Text>
      </group>
    );
  }

  return (
    <group>
      {/* Board surface */}
      <BoardSurface />

      {/* My area (bottom) */}
      <PlayerArea
        board={gameState.myBoard}
        hand={gameState.myHand}
        isOpponent={false}
      />

      {/* Opponent area (top) */}
      <PlayerArea board={gameState.opponentBoard} isOpponent={true} />
    </group>
  );
}
