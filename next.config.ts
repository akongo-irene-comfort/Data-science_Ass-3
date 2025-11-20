import type { NextConfig } from "next";
import path from "node:path";

const LOADER = path.resolve(__dirname, 'src/visual-edits/component-tagger-loader.js');

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: '**' },
      { protocol: 'http', hostname: '**' },
    ],
  },
  outputFileTracingRoot: path.resolve(__dirname, '../../'),
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    turbo: true, // enable Turbopack
  },
  webpack(config) {
    // Add custom loader safely
    config.module.rules.push({
      test: /\.(jsx|tsx)$/,
      use: LOADER,
    });
    return config;
  },
};

export default nextConfig;
