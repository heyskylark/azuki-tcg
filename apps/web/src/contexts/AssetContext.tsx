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

interface AssetContextValue {
  loadingState: AssetLoadingState;

  // Preload card images from deck data
  preloadDeckCards: (cards: DeckCard[]) => Promise<Map<string, CardMapping>>;

  // Get a preloaded texture by cardCode
  getCardTexture: (cardCode: string) => THREE.Texture | null;

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

  // Card back texture (created on client only)
  const [cardBackTexture, setCardBackTexture] = useState<THREE.Texture | null>(null);

  // Create card back texture on mount (client-side only)
  useEffect(() => {
    setCardBackTexture(createCardBackTexture());
  }, []);

  const preloadDeckCards = useCallback(
    async (cards: DeckCard[]): Promise<Map<string, CardMapping>> => {
      // Build unique card mappings
      const cardMappings = new Map<string, CardMapping>();
      const urlsToLoad: { cardCode: string; url: string }[] = [];

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
            urlsToLoad.push({ cardCode: card.cardCode, url: imageUrl });
          }
        }
      }

      if (urlsToLoad.length === 0) {
        // All textures already cached
        return cardMappings;
      }

      setLoadingState({
        isLoading: true,
        progress: 0,
        loadedCount: 0,
        totalCount: urlsToLoad.length,
        error: null,
      });

      const textureLoader = new THREE.TextureLoader();
      let loadedCount = 0;

      // Load all textures in parallel
      const loadPromises = urlsToLoad.map(
        ({ cardCode, url }) =>
          new Promise<void>((resolve) => {
            textureLoader.load(
              url,
              (texture) => {
                // Set proper color space for correct colors
                texture.colorSpace = THREE.SRGBColorSpace;
                textureCache.current.set(cardCode, texture);
                loadedCount++;
                setLoadingState((prev) => ({
                  ...prev,
                  loadedCount,
                  progress: Math.round((loadedCount / urlsToLoad.length) * 100),
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
                  progress: Math.round((loadedCount / urlsToLoad.length) * 100),
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

      return cardMappings;
    },
    []
  );

  const getCardTexture = useCallback((cardCode: string): THREE.Texture | null => {
    return textureCache.current.get(cardCode) ?? null;
  }, []);

  const clearAssets = useCallback(() => {
    // Dispose all cached textures
    for (const texture of textureCache.current.values()) {
      texture.dispose();
    }
    textureCache.current.clear();

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
        getCardTexture,
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
