import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * Runtime proxy: forwards /api/browgene/* → BROWGENE_API_URL/api/*
 * Unlike next.config.ts rewrites, this reads the env var at runtime,
 * so it works correctly inside Docker where the env is set via compose.
 */
const BROWGENE_API_URL = () =>
  process.env.BROWGENE_API_URL || 'http://localhost:8200';

async function proxyRequest(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const target = `${BROWGENE_API_URL()}/api/${path.join('/')}`;
  const url = new URL(target);

  // Forward query params
  const { searchParams } = new URL(req.url);
  searchParams.forEach((value, key) => url.searchParams.set(key, value));

  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };

    const fetchOpts: RequestInit = {
      method: req.method,
      headers,
      cache: 'no-store',
    };

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      fetchOpts.body = await req.text();
    }

    const response = await fetch(url.toString(), fetchOpts);
    const data = await response.text();

    return new NextResponse(data, {
      status: response.status,
      headers: { 'Content-Type': response.headers.get('Content-Type') || 'application/json' },
    });
  } catch (error) {
    console.error(`Proxy error [${req.method} ${url}]:`, error);
    return NextResponse.json(
      { error: 'Failed to connect to BrowGene API', details: String(error) },
      { status: 502 }
    );
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const DELETE = proxyRequest;
export const PATCH = proxyRequest;
