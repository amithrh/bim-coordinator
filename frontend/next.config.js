/** @type {import('next').NextConfig} */
const BACKEND = process.env.BIM_API_URL || "http://127.0.0.1:8009";
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
    ];
  },
};
module.exports = nextConfig;
