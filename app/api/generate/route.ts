// ============================================================
// app/api/generate/route.ts
// Mahwous Perfume AI — Image Generation Pipeline v15
//
// STRATEGY (v15): Two-Stage Pipeline — IMAGE STITCHING + KONTEXT
//
//   Stage 1: FLUX LoRA (text-to-image)
//     → Generates MAHWOUS_MAN character holding a generic perfume bottle
//     → LoRA weights guarantee consistent face/body
//     → Natural holding pose with full hand grip
//
//   Stage 2: Image Stitching + Kontext LoRA (image-to-image)
//     → Stitches Stage 1 image (LEFT) + real bottle photo (RIGHT)
//     → Sends stitched image to Kontext with prompt:
//       "Replace the bottle in the left image with the EXACT bottle
//        from the right image"
//     → Kontext transfers the real bottle design onto the character's hands
//     → Crops output to left half (character only)
//     → Result: character holding the REAL bottle naturally
//
// WHY THIS WORKS:
//   - Kontext can "see" the real bottle in the stitched image
//   - No text description needed — visual reference is pixel-perfect
//   - The character's hands naturally adapt to hold the real bottle
//   - Same technique used in professional product placement
//
// FALLBACK: If Stage 2 fails, Stage 1 image is returned as-is
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
// Stage 2: Image Stitching + Kontext LoRA
// 1. Download Stage 1 character image + real bottle image
// 2. Stitch them side by side (character LEFT, bottle RIGHT)
// 3. Send to Kontext with prompt to transfer the bottle
// 4. Crop output to left half (character with real bottle)
// ═══════════════════════════════════════════════════════════════════════════════

async function stitchAndTransferBottle(params: {
  characterImageUrl: string;
  bottleImageUrl: string;
  bottleName: string;
  bottleDescription?: string;
  loraPath: string;
  format: 'story' | 'post' | 'landscape';
  imageSize: { width: number; height: number };
}): Promise<string> {
  const { characterImageUrl, bottleImageUrl, bottleName, bottleDescription, loraPath, format, imageSize } = params;

  // ── Step 1: Download both images ──
  console.log(`[stitch] Downloading character image...`);
  const charBuffer = await fetchImageBuffer(characterImageUrl);
  if (!charBuffer) throw new Error('Failed to download character image');

  console.log(`[stitch] Downloading bottle image from: ${bottleImageUrl.substring(0, 80)}...`);
  const bottleBuffer = await fetchImageBuffer(bottleImageUrl);
  if (!bottleBuffer) throw new Error('Failed to download bottle image');

  // ── Step 2: Stitch images side by side ──
  // Get character dimensions
  const charMeta = await sharp(charBuffer).metadata();
  const charW = charMeta.width || imageSize.width;
  const charH = charMeta.height || imageSize.height;

  // Resize bottle to match character height (maintain aspect ratio)
  const bottleMeta = await sharp(bottleBuffer).metadata();
  const bottleOrigW = bottleMeta.width || 800;
  const bottleOrigH = bottleMeta.height || 800;
  const bottleNewH = charH;
  const bottleNewW = Math.round(bottleOrigW * (charH / bottleOrigH));

  const bottleResized = await sharp(bottleBuffer)
    .resize(bottleNewW, bottleNewH, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
    .jpeg({ quality: 90 })
    .toBuffer();

  // Ensure character is JPEG
  const charJpeg = await sharp(charBuffer)
    .jpeg({ quality: 95 })
    .toBuffer();

  // Create stitched canvas: character LEFT + bottle RIGHT
  const stitchW = charW + bottleNewW;
  const stitchH = charH;

  const stitchedBuffer = await sharp({
    create: {
      width: stitchW,
      height: stitchH,
      channels: 3,
      background: { r: 255, g: 255, b: 255 },
    },
  })
    .composite([
      { input: charJpeg, left: 0, top: 0 },
      { input: bottleResized, left: charW, top: 0 },
    ])
    .jpeg({ quality: 90 })
    .toBuffer();

  console.log(`[stitch] Stitched image: ${stitchW}x${stitchH} (${Math.round(stitchedBuffer.length / 1024)}KB)`);

  // Convert to base64 data URI
  const stitchedBase64 = `data:image/jpeg;base64,${stitchedBuffer.toString('base64')}`;

  // ── Step 3: Send to Kontext LoRA ──
  // Build a detailed prompt for bottle transfer
  const bottleDesc = bottleDescription
    ? `The bottle on the right is: ${bottleDescription}.`
    : `The bottle on the right is the ${bottleName} perfume bottle.`;

  const kontextPrompt = `The person on the left is holding a perfume bottle. Replace that bottle with the EXACT perfume bottle shown on the right side of the image. ${bottleDesc} The replacement must be pixel-perfect — same shape, same colors, same labels, same cap, same design details. Keep everything else unchanged — same person, same pose, same hands, same face, same background, same 3D Pixar animation style. Only change the bottle to match the reference on the right exactly. The character should naturally hold this exact bottle with both hands.`;

  console.log(`[stitch] Sending to Kontext LoRA for bottle transfer...`);

  const kontextInput: Record<string, unknown> = {
    image_url: stitchedBase64,
    prompt: kontextPrompt,
    num_inference_steps: 28,
    guidance_scale: 4.5,
    num_images: 1,
    enable_safety_checker: false,
    output_format: 'jpeg',
    loras: [{ path: loraPath.trim(), scale: 0.4 }],
  };

  const requestId = await submitToFal(FAL_MODEL_KONTEXT, kontextInput);
  const kontextResultUrl = await pollFalUntilDone(FAL_MODEL_KONTEXT, requestId, 120_000);

  console.log(`[stitch] Kontext result: ${kontextResultUrl.substring(0, 60)}...`);

  // ── Step 4: Download result and crop to left half (character only) ──
  const resultBuffer = await fetchImageBuffer(kontextResultUrl);
  if (!resultBuffer) throw new Error('Failed to download Kontext result');

  const resultMeta = await sharp(resultBuffer).metadata();
  const resultW = resultMeta.width || stitchW;
  const resultH = resultMeta.height || stitchH;

  // Calculate crop width proportional to original character width
  const cropW = Math.round(resultW * (charW / stitchW));

  const croppedBuffer = await sharp(resultBuffer)
    .extract({ left: 0, top: 0, width: cropW, height: resultH })
    .jpeg({ quality: 95 })
    .toBuffer();

  console.log(`[stitch] Cropped to character: ${cropW}x${resultH} (${Math.round(croppedBuffer.length / 1024)}KB)`);

  // ── Step 5: Upload cropped result to fal.ai storage ──
  const publicUrl = await uploadBufferToFal(croppedBuffer, `v15_${format}_${Date.now()}.jpg`);
  console.log(`[stitch] Final uploaded: ${publicUrl.substring(0, 80)}`);

  return publicUrl;
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

// ─── Generate a single format (Two-Stage Pipeline v15) ──────────────────────
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
    console.log(`[pipeline-v15] Generating ${ac.format} — "${perfumeData.name}" by ${perfumeData.brand}`);

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

    // ── Stage 2: Image Stitching + Kontext — Transfer real bottle ──
    const hasProductImage = productImageUrl && productImageUrl.trim().length > 10;

    if (hasProductImage) {
      try {
        const bottleName = `${perfumeData.brand} ${perfumeData.name}`.trim();
        console.log(`[stage-2] ${ac.format}: Image Stitching + Kontext — transferring real bottle...`);

        const finalUrl = await stitchAndTransferBottle({
          characterImageUrl: stage1Url,
          bottleImageUrl: productImageUrl!.trim(),
          bottleName,
          bottleDescription,
          loraPath,
          format: ac.format,
          imageSize: ac.imageSize,
        });

        console.log(`[stage-2] ${ac.format}: SUCCESS — ${finalUrl.substring(0, 60)}...`);

        return {
          format: ac.format,
          label: ac.label,
          dimensions: ac.dimensions,
          aspectRatio: ac.aspectRatio,
          url: finalUrl,
          status: 'COMPLETED',
          pipeline: 'stitch_kontext_v15',
        };
      } catch (stage2Err) {
        // Stage 2 failed — fall back to Stage 1 image
        console.warn(`[stage-2] ${ac.format}: Stitching+Kontext FAILED, using Stage 1 image:`, stage2Err);
        return {
          format: ac.format,
          label: ac.label,
          dimensions: ac.dimensions,
          aspectRatio: ac.aspectRatio,
          url: stage1Url,
          status: 'COMPLETED',
          pipeline: 'flux_lora_only_v15_fallback',
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
      pipeline: 'flux_lora_only_v15',
    };

  } catch (err) {
    console.error(`[pipeline-v15] FAILED (${ac.format}):`, err);
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

    // Use the real product image URL from scraper (perfumeData.imageUrl) for Stage 2
    const effectiveProductImageUrl =
      body.productImageUrl ||
      body.perfumeData?.imageUrl ||
      undefined;

    const enrichedBody = {
      ...body,
      bottleDescription: effectiveBottleDescription,
      productImageUrl: effectiveProductImageUrl,
    };

    console.log(`[generate] Pipeline v15 (Image Stitching + Kontext) — "${body.perfumeData.name}" by ${body.perfumeData.brand}`);
    console.log(`[generate] Product image URL: ${effectiveProductImageUrl ? effectiveProductImageUrl.substring(0, 80) : 'none — Stage 1 only'}`);
    console.log(`[generate] Pipeline mode: ${effectiveProductImageUrl ? 'STITCH + KONTEXT (real bottle transfer)' : 'STAGE 1 ONLY (no product image)'}`);

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
      pipeline: 'stitch_kontext_v15',
    }, { status: 200 });

  } catch (error: unknown) {
    console.error('[/api/generate] Error:', error);
    const message = error instanceof Error ? error.message : 'Image generation failed.';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
