"use client";

interface LoadingScreenProps {
  progress: number;
  message?: string;
}

/**
 * Loading screen with progress bar.
 * Displayed while card assets are being preloaded.
 */
export function LoadingScreen({
  progress,
  message = "Loading game assets...",
}: LoadingScreenProps) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 to-black">
      {/* Logo/Title */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Azuki TCG</h1>
        <p className="text-gray-400 text-center">{message}</p>
      </div>

      {/* Progress bar container */}
      <div className="w-80 max-w-[90%]">
        {/* Progress bar background */}
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
          {/* Progress bar fill */}
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Progress percentage */}
        <div className="mt-2 text-center">
          <span className="text-2xl font-bold text-white">{Math.round(progress)}%</span>
        </div>
      </div>

      {/* Loading tips (optional) */}
      <div className="mt-8 text-gray-500 text-sm max-w-md text-center px-4">
        <p>Tip: Drag to rotate the board, scroll to zoom</p>
      </div>

      {/* Animated loading indicator */}
      <div className="mt-8 flex space-x-2">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Mini loading indicator for inline use.
 */
export function LoadingSpinner({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizeClasses = {
    sm: "w-4 h-4 border-2",
    md: "w-8 h-8 border-2",
    lg: "w-12 h-12 border-4",
  };

  return (
    <div
      className={`${sizeClasses[size]} border-blue-500 border-t-transparent rounded-full animate-spin`}
    />
  );
}
