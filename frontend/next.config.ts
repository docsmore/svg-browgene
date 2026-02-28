import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/browgene/:path*",
        destination: `${process.env.BROWGENE_API_URL || "http://localhost:8200"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
