import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // API proxy is handled by app/api/browgene/[...path]/route.ts at runtime
  // (rewrites are baked at build time and don't pick up runtime env vars in Docker)
};

export default nextConfig;
