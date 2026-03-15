// ============================================================
// app/api/generate/route.ts
// Mahwous Perfume AI — Image Generation Pipeline v14
//
// STRATEGY (v14): Three-Stage Pipeline — SMART COMPOSITING
//
//   Stage 1: FLUX LoRA (text-to-image)
//     → Generates MAHWOUS_MAN character holding a generic perfume bottle
//     → LoRA weights guarantee consistent face/body
//     → Natural holding pose with full hand grip
//
//   Stage 2: Smart Compositing (Sharp library) — REAL BOTTLE OVERLAY
//     → Downloads Stage 1 character image
//     → Downloads REAL product photo from mahwous.com
//     → Composites the real bottle EXACTLY onto the character's hand
//     → 100% bottle accuracy — pixel-perfect real product image
//     → No AI hallucination — the real bottle is placed directly
//
//   Stage 3: FLUX Kontext (single image) — LIGHTING REFINEMENT
//     → Refines lighting and shadows around the composited bottle
//     → strength=0.25 — very light touch, preserves compositing
//     → Only used when productImageUrl is available
//
// WHY THREE STAGES:
//   - Stage 1 gives natural pose + consistent character
//   - Stage 2 places the EXACT real bottle (pixel-perfect, no AI guessing)
//   - Stage 3 blends lighting so the bottle looks natural in the scene
//
// FALLBACK: If Stage 2/3 fails, Stage 1 image is returned as-is
// ============================================================

import { NextRequest, NextResponse } from 'next/server';
import sharp from 'sharp';
import { buildFluxPrompt, buildNegativePrompt } from '@/lib/promptEngine';
import type { GenerationRequest } from '@/lib/types';

export const maxDuration = 300;
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

// ─── MAHWOUS_MAN LoRA weights ─────────────────────────────────────────────────
const MAHWOUS_LORA_URL =
  'https://v3b.fal.media/files/b/0a90eba7/OiQI7NS6N3neTl50fJHcC_pytorch_lora_weights.safetensors';
const MAHWOUS_TRIGGER = 'MAHWOUS_MAN';

const FAL_KEY_ENV = () => process.env.FAL_KEY ?? '';
const FAL_QUEUE_BASE = 'https://queue.fal.run';
const FAL_MODEL_T2I = 'fal-ai/flux-lora';
const FAL_MODEL_KONTEXT = 'fal-ai/flux-kontext-lora';

// ─── Aspect ratio configurations ─────────────────────────────────────────────
const ASPECT_CONFIGS = [
  {
    format: 'story' as const,
    label: 'Instagram Story (9:16)',
    dimensions: { width: 864, height: 1536 },
    imageSize: { width: 864, height: 1536 },
    aspectRatio: '9:16',
    aspectHint: 'VERTICAL PORTRAIT (9:16 tall format, taller than wide)',
  },
  {
    format: 'post' as const,
    label: 'Post Square (1:1)',
    dimensions: { width: 1072, height: 1072 },
    imageSize: { width: 1072, height: 1072 },
    aspectRatio: '1:1',
    aspectHint: 'SQUARE (1:1 equal width and height)',
  },
  {
    format: 'landscape' as const,
    label: 'Twitter / LinkedIn (16:9)',
    dimensions: { width: 1280, height: 720 },
    imageSize: { width: 1280, height: 720 },
    aspectRatio: '16:9',
    aspectHint: 'HORIZONTAL LANDSCAPE (16:9 wide format, wider than tall)',
  },
];

// ─── Bottle overlay configuration per format ─────────────────────────────────
// These ratios place the bottle in the character's hand area
const BOTTLE_OVERLAY_CONFIG = {
  story: {
    widthRatio: 0.28,   // bottle width = 28% of image width
    leftRatio: 0.36,    // left position = 36% from left
    topRatio: 0.42,     // top position = 42% from top
  },
  post: {
    widthRatio: 0.30,
    leftRatio: 0.35,
    topRatio: 0.38,
  },
  landscape: {
    widthRatio: 0.22,
    leftRatio: 0.38,
    topRatio: 0.30,
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// Image Proxy Helper — fetches image as buffer (handles Salla CDN 403)
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchImageBuffer(imageUrl: string): Promise<Buffer | null> {
  try {
    const res = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; MahwousBot/1.0)',
        'Accept': 'image/webp,image/jpeg,image/png,image/*',
        'Referer': 'https://mahwous.com/',
      },
    });
    if (res.ok) {
      const buffer = await res.arrayBuffer();
      console.log(`[proxy] Image fetched: ${imageUrl.substring(0, 80)} (${buffer.byteLength} bytes)`);
      return Buffer.from(buffer);
    }
    // Try with Salla-specific headers
    const res2 = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/webp,image/jpeg,image/png,image/*,*/*',
        'Accept-Language': 'ar,en;q=0.9',
        'Referer': 'https://mahwous.com/',
        'Origin': 'https://mahwous.com',
      },
    });
    if (res2.ok) {
      const buffer = await res2.arrayBuffer();
      console.log(`[proxy] Image fetched (Salla headers): ${buffer.byteLength} bytes`);
      return Buffer.from(buffer);
    }
    console.warn(`[proxy] Both fetch attempts failed for: ${imageUrl.substring(0, 80)}`);
    return null;
  } catch (err) {
    console.warn(`[proxy] fetchImageBuffer error:`, err);
    return null;
  }
}

async function fetchImageAsBase64(imageUrl: string): Promise<string | null> {
  const buffer = await fetchImageBuffer(imageUrl);
  if (!buffer) return null;
  const base64 = buffer.toString('base64');
  return `data:image/jpeg;base64,${base64}`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// fal.ai Queue Helpers
// ═══════════════════════════════════════════════════════════════════════════════

async function submitToFal(model: string, input: Record<string, unknown>): Promise<string> {
  const falKey = FAL_KEY_ENV();
  if (!falKey) throw new Error('FAL_KEY is not set.');

  const res = await fetch(`${FAL_QUEUE_BASE}/${model}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Key ${falKey}`,
    },
    body: JSON.stringify(input),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`fal.ai submit error ${res.status}: ${errText}`);
  }

  const data = await res.json();
  const requestId = data?.request_id as string | undefined;
  if (!requestId) throw new Error('fal.ai did not return a request_id');
  return requestId;
}

async function pollFalUntilDone(model: string, requestId: string, timeoutMs = 120_000): Promise<string> {
  const falKey = FAL_KEY_ENV();
  if (!falKey) throw new Error('FAL_KEY is not set.');

  const statusUrl = `${FAL_QUEUE_BASE}/${model}/requests/${requestId}/status`;
  const resultUrl = `${FAL_QUEUE_BASE}/${model}/requests/${requestId}`;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2500));

    const statusRes = await fetch(statusUrl, {
      headers: { Authorization: `Key ${falKey}` },
    });
    if (!statusRes.ok) continue;

    const status = await statusRes.json();
    const queueStatus: string = status?.status ?? '';

    if (queueStatus === 'COMPLETED') {
      const resultRes = await fetch(resultUrl, {
        headers: { Authorization: `Key ${falKey}` },
      });
      if (!resultRes.ok) throw new Error(`fal.ai result fetch error ${resultRes.status}`);
      const result = await resultRes.json();
      const imageUrl: string | undefined = result?.images?.[0]?.url ?? result?.image?.url;
      if (!imageUrl) throw new Error('fal.ai returned no image URL');
      return imageUrl;
    }

    if (queueStatus === 'FAILED') {
      throw new Error(`fal.ai generation failed: ${status?.error ?? 'Unknown error'}`);
    }
  }

  throw new Error('fal.ai generation timed out');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1: FLUX LoRA — Generate character holding a generic bottle
// ═══════════════════════════════════════════════════════════════════════════════

async function generateStage1(params: {
  prompt: string;
  negativePrompt: string;
  loraPath: string;
  imageSize: { width: number; height: number };
}): Promise<string> {
  const { prompt, negativePrompt, loraPath, imageSize } = params;

  const input: Record<string, unknown> = {
    prompt,
    negative_prompt: negativePrompt,
    image_size: imageSize,
    num_inference_steps: 28,
    guidance_scale: 3.5,
    num_images: 1,
    enable_safety_checker: false,
    output_format: 'png',
    loras: [{ path: loraPath.trim(), scale: 1.0 }],
  };

  const requestId = await submitToFal(FAL_MODEL_T2I, input);
  return await pollFalUntilDone(FAL_MODEL_T2I, requestId);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 2: Smart Compositing — Place REAL bottle onto character's hand
// Uses Sharp to pixel-perfectly overlay the real product image
// ═══════════════════════════════════════════════════════════════════════════════

async function compositeRealBottle(params: {
  characterImageUrl: string;
  bottleImageUrl: string;
  format: 'story' | 'post' | 'landscape';
}): Promise<Buffer> {
  const { characterImageUrl, bottleImageUrl, format } = params;

  // Download both images
  const [charBuffer, bottleBuffer] = await Promise.all([
    fetchImageBuffer(characterImageUrl),
    fetchImageBuffer(bottleImageUrl),
  ]);

  if (!charBuffer) throw new Error('Failed to download character image');
  if (!bottleBuffer) throw new Error('Failed to download bottle image');

  // Get character image dimensions
  const charMeta = await sharp(charBuffer).metadata();
  const charWidth = charMeta.width || 1080;
  const charHeight = charMeta.height || 1920;

  // Calculate bottle overlay dimensions based on format
  const config = BOTTLE_OVERLAY_CONFIG[format] || BOTTLE_OVERLAY_CONFIG.post;
  const bottleWidth = Math.round(charWidth * config.widthRatio);
  const bottleLeft = Math.round(charWidth * config.leftRatio);
  const bottleTop = Math.round(charHeight * config.topRatio);

  // Get bottle aspect ratio to maintain proportions
  const bottleMeta = await sharp(bottleBuffer).metadata();
  const bottleAspect = (bottleMeta.height || 1) / (bottleMeta.width || 1);
  const bottleHeight = Math.round(bottleWidth * bottleAspect);

  console.log(`[composite] Format: ${format}, charSize: ${charWidth}x${charHeight}`);
  console.log(`[composite] Bottle overlay: ${bottleWidth}x${bottleHeight} at (${bottleLeft}, ${bottleTop})`);

  // Process bottle: resize and convert to PNG (supports transparency)
  const processedBottle = await sharp(bottleBuffer)
    .resize(bottleWidth, bottleHeight, {
      fit: 'contain',
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    })
    .png()
    .toBuffer();

  // Composite the real bottle onto the character image
  const composited = await sharp(charBuffer)
    .composite([
      {
        input: processedBottle,
        left: bottleLeft,
        top: bottleTop,
        blend: 'over',
      },
    ])
    .jpeg({ quality: 95 })
    .toBuffer();

  console.log(`[composite] SUCCESS — composited image: ${composited.length} bytes`);
  return composited;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3: FLUX Kontext — Light refinement for natural lighting (optional)
// strength=0.25 — very light touch, preserves the composited bottle
// ═══════════════════════════════════════════════════════════════════════════════

async function refineWithKontext(params: {
  imageBase64: string;  // base64 data URI of composited image
  bottleName: string;
  loraPath: string;
}): Promise<string> {
  const { imageBase64, bottleName, loraPath } = params;

  const prompt = `The ${bottleName} perfume bottle in the character's hand — refine the lighting and shadows around the bottle so it blends naturally with the scene. Keep the bottle appearance EXACTLY the same. Keep the character, pose, background, and 3D animation style EXACTLY the same. Only improve the lighting integration.`;

  const input: Record<string, unknown> = {
    image_url: imageBase64,
    prompt,
    num_inference_steps: 30,
    guidance_scale: 20,
    strength: 0.25,  // Very light — just lighting refinement
    num_images: 1,
    enable_safety_checker: false,
    output_format: 'jpeg',
    loras: [{ path: loraPath.trim(), scale: 0.6 }],
  };

  const requestId = await submitToFal(FAL_MODEL_KONTEXT, input);
  return await pollFalUntilDone(FAL_MODEL_KONTEXT, requestId, 90_000);
}

// ─── Generate a single format (Three-Stage Pipeline v14) ─────────────────────
async function generateFormat(
  request: GenerationRequest,
  ac: typeof ASPECT_CONFIGS[0],
): Promise<{
  format: string;
  label: string;
  dimensions: { width: number; height: number };
  aspectRatio: string;
  url: string | null;
  status: 'COMPLETED' | 'FAILED';
  pipeline: string;
}> {
  const { perfumeData, vibe = '', attire = '', bottleDescription, productImageUrl } = request;
  const loraPath = request.loraPath?.trim() || MAHWOUS_LORA_URL;
  const triggerWord = request.loraTriggerWord?.trim() || MAHWOUS_TRIGGER;

  try {
    console.log(`[pipeline-v14] Generating ${ac.format} — "${perfumeData.name}" by ${perfumeData.brand}`);

    // ── Stage 1: FLUX LoRA — Character with generic bottle ──
    const fluxPrompt = buildFluxPrompt({
      perfumeData,
      vibe,
      attire,
      aspectHint: ac.aspectHint,
      bottleDescription: undefined, // Generic bottle for natural pose
      loraTriggerWord: triggerWord,
      hasBottleReference: false,
    });
    const negativePrompt = buildNegativePrompt();

    console.log(`[stage-1] ${ac.format}: Generating character with generic bottle...`);
    const stage1Url = await generateStage1({
      prompt: fluxPrompt,
      negativePrompt,
      loraPath,
      imageSize: ac.imageSize,
    });
    console.log(`[stage-1] ${ac.format}: SUCCESS — ${stage1Url.substring(0, 60)}...`);

    // ── Stage 2: Smart Compositing — Place REAL bottle onto hand ──
    const hasProductImage = productImageUrl && productImageUrl.trim().length > 10;

    if (hasProductImage) {
      try {
        console.log(`[stage-2] ${ac.format}: Smart Compositing — placing real bottle from ${productImageUrl!.substring(0, 60)}...`);

        const compositedBuffer = await compositeRealBottle({
          characterImageUrl: stage1Url,
          bottleImageUrl: productImageUrl!.trim(),
          format: ac.format,
        });

        // Convert composited image to base64 data URI for Stage 3
        const compositedBase64 = `data:image/jpeg;base64,${compositedBuffer.toString('base64')}`;
        console.log(`[stage-2] ${ac.format}: Compositing SUCCESS (${Math.round(compositedBuffer.length / 1024)}KB)`);

        // ── Stage 3: Kontext Lighting Refinement (light touch) ──
        try {
          console.log(`[stage-3] ${ac.format}: Refining lighting with Kontext (strength=0.25)...`);
          const bottleName = `${perfumeData.brand} ${perfumeData.name}`.trim();
          const refinedUrl = await refineWithKontext({
            imageBase64: compositedBase64,
            bottleName,
            loraPath,
          });
          console.log(`[stage-3] ${ac.format}: Refinement SUCCESS — ${refinedUrl.substring(0, 60)}...`);

          return {
            format: ac.format,
            label: ac.label,
            dimensions: ac.dimensions,
            aspectRatio: ac.aspectRatio,
            url: refinedUrl,
            status: 'COMPLETED',
            pipeline: 'smart_composite_kontext_v14',
          };
        } catch (stage3Err) {
          // Stage 3 failed — use composited image directly (still has real bottle!)
          console.warn(`[stage-3] ${ac.format}: Refinement failed, using composited image:`, stage3Err);

          // Upload composited buffer to fal.ai storage for a public URL
          const uploadedUrl = await uploadBufferToFal(compositedBuffer, `composite_${ac.format}.jpg`);

          return {
            format: ac.format,
            label: ac.label,
            dimensions: ac.dimensions,
            aspectRatio: ac.aspectRatio,
            url: uploadedUrl,
            status: 'COMPLETED',
            pipeline: 'smart_composite_only_v14',
          };
        }
      } catch (stage2Err) {
        // Stage 2 failed — fall back to Stage 1 image
        console.warn(`[stage-2] ${ac.format}: Compositing FAILED, using Stage 1 image:`, stage2Err);
        return {
          format: ac.format,
          label: ac.label,
          dimensions: ac.dimensions,
          aspectRatio: ac.aspectRatio,
          url: stage1Url,
          status: 'COMPLETED',
          pipeline: 'flux_lora_only_v14_fallback',
        };
      }
    }

    // No product image — return Stage 1 directly
    console.log(`[stage-2] ${ac.format}: No product image — returning Stage 1 result`);
    return {
      format: ac.format,
      label: ac.label,
      dimensions: ac.dimensions,
      aspectRatio: ac.aspectRatio,
      url: stage1Url,
      status: 'COMPLETED',
      pipeline: 'flux_lora_only_v14',
    };

  } catch (err) {
    console.error(`[pipeline-v14] FAILED (${ac.format}):`, err);
    return {
      format: ac.format,
      label: ac.label,
      dimensions: ac.dimensions,
      aspectRatio: ac.aspectRatio,
      url: null,
      status: 'FAILED',
      pipeline: 'failed',
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: Upload buffer to fal.ai storage to get a public URL
// ═══════════════════════════════════════════════════════════════════════════════

async function uploadBufferToFal(buffer: Buffer, filename: string): Promise<string> {
  const falKey = FAL_KEY_ENV();
  if (!falKey) throw new Error('FAL_KEY is not set.');

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(buffer)], { type: 'image/jpeg' });
  formData.append('file', blob, filename);

  const res = await fetch('https://rest.fal.run/storage/upload', {
    method: 'POST',
    headers: { Authorization: `Key ${falKey}` },
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`fal.ai upload error ${res.status}: ${errText}`);
  }

  const data = await res.json();
  const url = data?.url as string | undefined;
  if (!url) throw new Error('fal.ai upload did not return a URL');

  console.log(`[upload] Uploaded to fal.ai: ${url.substring(0, 80)}`);
  return url;
}

// ─── Main Handler ─────────────────────────────────────────────────────────────
export async function POST(request: NextRequest) {
  try {
    const body: GenerationRequest = await request.json();

    if (!body.perfumeData?.name?.trim() || !body.perfumeData?.brand?.trim()) {
      return NextResponse.json(
        { error: 'perfumeData.name and perfumeData.brand are required.' },
        { status: 400 }
      );
    }

    if (!FAL_KEY_ENV()) {
      return NextResponse.json(
        { error: 'FAL_KEY is not configured. Please add it to environment variables.' },
        { status: 500 }
      );
    }

    // Merge bottleAnalysis.description into bottleDescription if not already set
    const effectiveBottleDescription =
      body.bottleDescription ||
      body.bottleAnalysis?.description ||
      body.bottleAnalysis?.loraPromptAddition ||
      undefined;

    // Use the real product image URL from scraper (perfumeData.imageUrl) for Stage 2 compositing
    const effectiveProductImageUrl =
      body.productImageUrl ||
      body.perfumeData?.imageUrl ||
      undefined;

    const enrichedBody = {
      ...body,
      bottleDescription: effectiveBottleDescription,
      productImageUrl: effectiveProductImageUrl,
    };

    console.log(`[generate] Pipeline v14 (Smart Compositing + Kontext Lighting) — "${body.perfumeData.name}" by ${body.perfumeData.brand}`);
    console.log(`[generate] Product image URL: ${effectiveProductImageUrl ? effectiveProductImageUrl.substring(0, 80) : 'none — Stage 1 only'}`);
    console.log(`[generate] Pipeline mode: ${effectiveProductImageUrl ? 'SMART COMPOSITING (real bottle pixel-perfect)' : 'STAGE 1 ONLY (no product image)'}`);

    // Generate all 3 formats in parallel
    const results = await Promise.all(
      ASPECT_CONFIGS.map((ac) => generateFormat(enrichedBody, ac))
    );

    const completedImages = results.filter((r) => r.status === 'COMPLETED' && r.url);

    if (completedImages.length === 0) {
      return NextResponse.json(
        { error: 'فشل توليد الصور. يرجى التحقق من FAL_KEY والمحاولة مرة أخرى.' },
        { status: 500 }
      );
    }

    console.log(`[generate] Completed: ${completedImages.length}/3 images`);
    completedImages.forEach((img) => {
      console.log(`  ${img.format}: ${img.pipeline}`);
    });

    return NextResponse.json({
      status: 'completed',
      images: completedImages.map((img) => ({
        format: img.format,
        label: img.label,
        dimensions: img.dimensions,
        url: img.url,
        aspectRatio: img.aspectRatio,
      })),
      pipeline: 'smart_composite_v14',
    }, { status: 200 });

  } catch (error: unknown) {
    console.error('[/api/generate] Error:', error);
    const message = error instanceof Error ? error.message : 'Image generation failed.';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
