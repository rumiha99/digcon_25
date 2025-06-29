// clip.js
// 実行前に下記を実行
// npm install canvas onnxruntime-web
// メイン処理に画像ファイルパスを指定して実行

const fs       = require('fs').promises;
const { createCanvas, loadImage } = require('canvas');
const ort      = require('onnxruntime-web');

let onnxImageSession;  // モデル読み込み後にセットされます

/**
 * Web 上の ONNX モデルをダウンロードしてセッションを作成する
 */
async function loadImageModelFromWeb(url) {
  console.log(`Downloading model from ${url}...`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`✖ Model download failed: ${res.status} ${res.statusText}`);
  }
  const contentLength = res.headers.get('content-length');
  if (contentLength) {
    console.log(`↳ Expected size: ${contentLength} bytes`);
  }
  const arrayBuffer = await res.arrayBuffer();
  console.log(`✔ Downloaded: ${arrayBuffer.byteLength} bytes`);

  console.log("Creating ONNX session...");
  const session = await ort.InferenceSession.create(arrayBuffer, {
    executionProviders: ["wasm"]
  });
  console.log("✔ ONNX session created successfully");
  return session;
}

/**
 * 画像ファイルパスを入力として CLIP 画像エンコーダの
 * 埋め込みベクトルを計算する
 * @param {string} imagePath - './img1.png' のようなローカルファイルパス
 * @returns {Promise<Float32Array>}
 */
async function embedImageFromPath(imagePath) {
  const rgbData = await getRgbData(imagePath);
  const inputTensor = new ort.Tensor('float32', rgbData, [1, 3, 224, 224]);

  console.log(`Running inference on "${imagePath}"...`);
  const t0 = Date.now();
  const results = await onnxImageSession.run({ input: inputTensor });
  console.log(`Finished inference in ${Date.now() - t0} ms`);
  return results.output.data;
}

/**
 * ファイルパス → Buffer → loadImage(buffer) → Canvas 前処理
 * CLIP リポジトリ準拠の正規化を行う
 * @param {string} imgPath - ローカルファイルパス
 * @returns {Promise<Float32Array>} 長さ 3*224*224 の正規化済み配列
 */
async function getRgbData(imgPath) {
  // 1) ファイルを Buffer として読む
  let buffer;
  try {
    buffer = await fs.readFile(imgPath);
  } catch (e) {
    throw new Error(`Image file not found or unreadable: ${imgPath}`);
  }

  // 2) loadImage に Buffer を渡す（日本語パスも OK）
  const img = await loadImage(buffer);

  // 3) Canvas でリサイズ＆ピクセル取得
  const canvas = createCanvas(224, 224);
  const ctx    = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 224, 224);
  const data   = ctx.getImageData(0, 0, 224, 224).data;

  // CLIP の mean/std
  const MEAN = [0.48145466, 0.4578275, 0.40821073];
  const STD  = [0.26862954, 0.26130258, 0.27577711];

  // チャンネルごとに正規化して格納
  const size = 224 * 224;
  const r = new Float32Array(size);
  const g = new Float32Array(size);
  const b = new Float32Array(size);

  for (let i = 0, px = 0; i < data.length; i += 4, px++) {
    r[px] = (data[i + 0] / 255 - MEAN[0]) / STD[0];
    g[px] = (data[i + 1] / 255 - MEAN[1]) / STD[1];
    b[px] = (data[i + 2] / 255 - MEAN[2]) / STD[2];
  }

  // [r, g, b] を連結して 1 次元配列に
  const all = new Float32Array(3 * size);
  all.set(r, 0);
  all.set(g, size);
  all.set(b, 2 * size);
  return all;
}

/**
 * 2 つのローカル画像ファイル間でコサイン類似度を計算する
 * @param {string} pathA - 1枚目の画像パス
 * @param {string} pathB - 2枚目の画像パス
 * @returns {Promise<number>} 類似度スコア（-1..1）
 */
async function culCosineSimilarity(pathA, pathB) {
  const vecA = await embedImageFromPath(pathA);
  const vecB = await embedImageFromPath(pathB);

  if (vecA.length !== vecB.length) {
    throw new Error("Embedding vectors must have the same length");
  }

  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot   += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  const similarity = (normA === 0 || normB === 0) ? 0 : dot / (normA * normB);

  console.log(`Cosine similarity between "${pathA}" and "${pathB}":`, similarity);
  return similarity;
}

// ----------------------
// メイン処理
// ----------------------
(async () => {
  try {
    const modelUrl = "https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-float32.onnx";
    console.log("Loading CLIP model...");
    onnxImageSession = await loadImageModelFromWeb(modelUrl);

    console.log("Model loaded. Computing similarity...");
    // ここにローカル画像ファイルパスを指定
    const imagePath1 = "./sample1.png";
    const imagePath2 = "./sample2.png";
    const score = await culCosineSimilarity(imagePath1, imagePath2);

    console.log("Final similarity score:", score);
  } catch (err) {
    console.error("❌ Error in processing:", err);
  }
})();
