"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from "react";
import * as THREE from "three";
import type { AssetLoadingState, DeckCard, CardMapping } from "@/types/game";
import { buildImageUrl } from "@/types/game";
import type { SnapshotCardMetadata } from "@tcg/backend-core/types/ws";

interface AssetContextValue {
  loadingState: AssetLoadingState;

  // Preload card images from deck data
  preloadDeckCards: (cards: DeckCard[]) => Promise<Map<string, CardMapping>>;

  // Preload card images from snapshot metadata
  preloadCardsByMetadata: (
    metadata: Record<string, SnapshotCardMetadata>
  ) => Promise<Map<string, CardMapping>>;

  // Get a preloaded texture by cardCode
  getCardTexture: (cardCode: string) => THREE.Texture | null;

  // Get a preloaded UI texture by key
  getUiTexture: (key: string) => THREE.Texture | null;

  // Card back texture (always available after mount)
  cardBackTexture: THREE.Texture | null;

  // Check if assets are ready
  isReady: boolean;

  // Reset/clear assets
  clearAssets: () => void;
}

const AssetContext = createContext<AssetContextValue | null>(null);

interface AssetProviderProps {
  children: ReactNode;
}

const UI_TEXTURES = [
  {
    key: "target",
    url: "https://azuki-tcg.s3.us-east-1.amazonaws.com/textures/target.png",
  },
  {
    key: "target-pointer",
    url: "https://azuki-tcg.s3.us-east-1.amazonaws.com/textures/target-pointer.png",
  },
];

type TextureCacheKey = "card" | "ui";

interface TextureLoadEntry {
  key: string;
  url: string;
  cache: TextureCacheKey;
}

export function AssetProvider({ children }: AssetProviderProps) {
  const [loadingState, setLoadingState] = useState<AssetLoadingState>({
    isLoading: false,
    progress: 0,
    loadedCount: 0,
    totalCount: 0,
    error: null,
  });

  // Texture cache: cardCode -> THREE.Texture
  const textureCache = useRef<Map<string, THREE.Texture>>(new Map());
  // UI texture cache: key -> THREE.Texture
  const uiTextureCache = useRef<Map<string, THREE.Texture>>(new Map());

  // Card back texture (created on client only)
  const [cardBackTexture, setCardBackTexture] = useState<THREE.Texture | null>(null);

  // Create card back texture on mount (client-side only)
  useEffect(() => {
    setCardBackTexture(createCardBackTexture());
  }, []);

  const loadTextures = useCallback(
    async (entries: TextureLoadEntry[]) => {
      if (entries.length === 0) {
        return;
      }

      setLoadingState({
        isLoading: true,
        progress: 0,
        loadedCount: 0,
        totalCount: entries.length,
        error: null,
      });

      const textureLoader = new THREE.TextureLoader();
      let loadedCount = 0;

      const loadPromises = entries.map(
        ({ key, url, cache }) =>
          new Promise<void>((resolve) => {
            textureLoader.load(
              url,
              (texture) => {
                // Set proper color space for correct colors
                texture.colorSpace = THREE.SRGBColorSpace;
                if (cache === "card") {
                  textureCache.current.set(key, texture);
                } else {
                  uiTextureCache.current.set(key, texture);
                }
                loadedCount++;
                setLoadingState((prev) => ({
                  ...prev,
                  loadedCount,
                  progress: Math.round((loadedCount / entries.length) * 100),
                }));
                resolve();
              },
              undefined,
              () => {
                // On error, still count as loaded but don't cache
                loadedCount++;
                setLoadingState((prev) => ({
                  ...prev,
                  loadedCount,
                  progress: Math.round((loadedCount / entries.length) * 100),
                }));
                resolve();
              }
            );
          })
      );

      await Promise.all(loadPromises);

      setLoadingState((prev) => ({
        ...prev,
        isLoading: false,
        progress: 100,
      }));
    },
    []
  );

  const getUiTextureEntries = useCallback((): TextureLoadEntry[] => {
    return UI_TEXTURES.filter(
      (texture) => !uiTextureCache.current.has(texture.key)
    ).map((texture) => ({
      key: texture.key,
      url: texture.url,
      cache: "ui" as const,
    }));
  }, []);

  const preloadDeckCards = useCallback(
    async (cards: DeckCard[]): Promise<Map<string, CardMapping>> => {
      // Build unique card mappings
      const cardMappings = new Map<string, CardMapping>();
      const entriesToLoad: TextureLoadEntry[] = [];

      for (const card of cards) {
        if (!cardMappings.has(card.cardCode)) {
          const imageUrl = buildImageUrl(card.imageKey);
          cardMappings.set(card.cardCode, {
            cardCode: card.cardCode,
            imageKey: card.imageKey,
            imageUrl,
            name: card.name,
            cardType: card.cardType,
            attack: card.attack,
            health: card.health,
            ikzCost: card.ikzCost,
          });

          // Only load if not already cached
          if (!textureCache.current.has(card.cardCode)) {
            entriesToLoad.push({
              key: card.cardCode,
              url: imageUrl,
              cache: "card",
            });
          }
        }
      }

      entriesToLoad.push(...getUiTextureEntries());

      if (entriesToLoad.length > 0) {
        await loadTextures(entriesToLoad);
      }

      return cardMappings;
    },
    [getUiTextureEntries, loadTextures]
  );

  const preloadCardsByMetadata = useCallback(
    async (
      metadata: Record<string, SnapshotCardMetadata>
    ): Promise<Map<string, CardMapping>> => {
      const cardMappings = new Map<string, CardMapping>();
      const entriesToLoad: TextureLoadEntry[] = [];

      for (const cardMetadata of Object.values(metadata)) {
        const cardCode = cardMetadata.cardCode;
        if (!cardMappings.has(cardCode)) {
          const imageUrl = buildImageUrl(cardMetadata.imageKey);
          cardMappings.set(cardCode, {
            cardCode,
            imageKey: cardMetadata.imageKey,
            imageUrl,
            name: cardMetadata.name,
            cardType: cardMetadata.cardType,
            attack: cardMetadata.attack,
            health: cardMetadata.health,
            ikzCost: cardMetadata.ikzCost,
          });

          if (!textureCache.current.has(cardCode)) {
            entriesToLoad.push({
              key: cardCode,
              url: imageUrl,
              cache: "card",
            });
          }
        }
      }

      entriesToLoad.push(...getUiTextureEntries());

      if (entriesToLoad.length > 0) {
        await loadTextures(entriesToLoad);
      }

      return cardMappings;
    },
    [getUiTextureEntries, loadTextures]
  );

  const getCardTexture = useCallback((cardCode: string): THREE.Texture | null => {
    return textureCache.current.get(cardCode) ?? null;
  }, []);

  const getUiTexture = useCallback((key: string): THREE.Texture | null => {
    return uiTextureCache.current.get(key) ?? null;
  }, []);

  const clearAssets = useCallback(() => {
    // Dispose all cached textures
    for (const texture of textureCache.current.values()) {
      texture.dispose();
    }
    textureCache.current.clear();
    for (const texture of uiTextureCache.current.values()) {
      texture.dispose();
    }
    uiTextureCache.current.clear();

    setLoadingState({
      isLoading: false,
      progress: 0,
      loadedCount: 0,
      totalCount: 0,
      error: null,
    });
  }, []);

  const isReady = !loadingState.isLoading && loadingState.error === null;

  return (
    <AssetContext.Provider
      value={{
        loadingState,
        preloadDeckCards,
        preloadCardsByMetadata,
        getCardTexture,
        getUiTexture,
        cardBackTexture,
        isReady,
        clearAssets,
      }}
    >
      {children}
    </AssetContext.Provider>
  );
}

export function useAssets() {
  const context = useContext(AssetContext);
  if (!context) {
    throw new Error("useAssets must be used within an AssetProvider");
  }
  return context;
}

// ============================================
// Helper functions
// ============================================

/**
 * Create a simple card back texture (solid color for now).
 * This will be replaced with an actual card back image later.
 */
function createCardBackTexture(): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 358; // Roughly card aspect ratio

  const ctx = canvas.getContext("2d");
  if (ctx) {
    // Dark blue background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add a simple border
    ctx.strokeStyle = "#4a4a6e";
    ctx.lineWidth = 8;
    ctx.strokeRect(8, 8, canvas.width - 16, canvas.height - 16);

    // Add a simple pattern
    ctx.fillStyle = "#2a2a4e";
    const patternSize = 32;
    for (let x = 24; x < canvas.width - 24; x += patternSize) {
      for (let y = 24; y < canvas.height - 24; y += patternSize) {
        if ((x + y) % (patternSize * 2) === 0) {
          ctx.fillRect(x, y, patternSize - 4, patternSize - 4);
        }
      }
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
