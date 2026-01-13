import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  transpilePackages: ["@tcg/backend-core"],
  webpack: (config) => {
    // Add alias for @tcg/backend-core's internal @core/* imports
    config.resolve.alias = {
      ...config.resolve.alias,
      "@core/database": path.resolve(__dirname, "../../packages/backend-core/src/database"),
      "@core/drizzle": path.resolve(__dirname, "../../packages/backend-core/src/drizzle"),
      "@core/types": path.resolve(__dirname, "../../packages/backend-core/src/types"),
      "@core/constants": path.resolve(__dirname, "../../packages/backend-core/src/constants"),
      "@core/errors": path.resolve(__dirname, "../../packages/backend-core/src/errors"),
      "@core/utils": path.resolve(__dirname, "../../packages/backend-core/src/utils"),
      "@core/services": path.resolve(__dirname, "../../packages/backend-core/src/services"),
    };
    return config;
  },
};

export default nextConfig;
