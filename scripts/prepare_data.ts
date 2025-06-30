import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { encoding_for_model } from 'tiktoken';
import { ChromaClient } from 'chromadb';
import { pipeline } from '@xenova/transformers';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const OBJECTS_DIR = path.join(__dirname, '../objects/objects');

interface ArtObject {
  id: string;
  title?: string;
  description?: string;
  text?: string;
  artist?: string;
  culture?: string;
  style?: string;
  dated?: string;
  country?: string;
  medium?: string;
}

const CHUNK_SIZE = 128;
const CHUNK_OVERLAP = 32;
const TOKENIZER_MODEL = 'gpt-3.5-turbo';

const chroma = new ChromaClient();

const COLLECTION_NAME = 'artsmia_chunks';

let extractor: any;
async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractor;
}

async function getOrCreateCollection() {
  const collections = await chroma.listCollections();
  let exists = collections.includes(COLLECTION_NAME);
  if (!exists) {
    await chroma.createCollection({ name: COLLECTION_NAME, metadata: { dimension: 384 } });
  }
  return chroma.getCollection({
    name: COLLECTION_NAME,
    embeddingFunction: {} as unknown as any,
  });
}

function getAllJsonFiles(dir: string): string[] {
  let results: string[] = [];
  const list = fs.readdirSync(dir);
  list.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat && stat.isDirectory()) {
      results = results.concat(getAllJsonFiles(filePath));
    } else if (file.endsWith('.json')) {
      results.push(filePath);
    }
  });
  return results;
}

function loadArtObject(filePath: string): ArtObject | null {
  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    if (!raw.trim()) return null;
    const data = JSON.parse(raw);
    return {
      id: data.id || '',
      title: data.title,
      description: data.description,
      text: data.text,
      artist: data.artist,
      culture: data.culture,
      style: data.style,
      dated: data.dated,
      country: data.country,
      medium: data.medium,
    };
  } catch (e) {
    console.error('Error loading', filePath, e);
    return null;
  }
}

function buildChunkText(obj: ArtObject): string {
  return [
    obj.title,
    obj.description,
    obj.text,
    obj.artist,
    obj.culture,
    obj.style,
    obj.dated,
    obj.country,
    obj.medium,
  ]
    .filter(Boolean)
    .map((x: any) => {
      if (typeof x === 'string') return x;
      if (Buffer.isBuffer(x)) return x.toString('utf-8');
      if (ArrayBuffer.isView(x) && x.constructor && x.constructor.name === 'Uint8Array')
        return Buffer.from(new Uint8Array(x.buffer)).toString('utf-8');
      return String(x);
    })
    .join('\n');
}

function chunkTextTiktoken(
  text: string
): Array<{ text: string; start: number; end: number; tokens: number }> {
  const enc = encoding_for_model(TOKENIZER_MODEL);
  const tokens = enc.encode(text);
  const chunks: Array<{ text: string; start: number; end: number; tokens: number }> = [];
  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + CHUNK_SIZE, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    let chunkText = Buffer.from(enc.decode(chunkTokens)).toString('utf-8');
    chunks.push({
      text: chunkText,
      start,
      end,
      tokens: chunkTokens.length,
    });
    if (end === tokens.length) break;
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  enc.free();
  return chunks;
}

async function embedChunk(text: string): Promise<number[]> {
  if (typeof text !== 'string') {
    throw new Error('embedChunk: input is not a string');
  }
  if (text.length === 0) {
    return [];
  }
  const extractor = await getExtractor();
  const output = await extractor(text);
  let embedding: any = output.data;
  const DIM = 384; // all-MiniLM-L6-v2 output dim

  if (embedding instanceof Float32Array && embedding.length % DIM === 0) {
    const tokens = embedding.length / DIM;
    const arr = Array.from(embedding);
    const pooled = new Array(DIM).fill(0);
    for (let i = 0; i < tokens; i++) {
      for (let j = 0; j < DIM; j++) {
        pooled[j] += arr[i * DIM + j];
      }
    }
    for (let j = 0; j < DIM; j++) {
      pooled[j] /= tokens;
    }
    embedding = pooled;
  } else if (Array.isArray(embedding) && Array.isArray(embedding[0])) {
    const arr = embedding as ArrayLike<ArrayLike<number> | Float32Array>;
    const dim = arr[0].length;
    const sum = new Array(dim).fill(0);
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < dim; j++) {
        sum[j] += arr[i][j];
      }
      count++;
    }
    embedding = sum.map((x) => x / count);
  } else if (Array.isArray(embedding) && typeof embedding[0] === 'number') {
    embedding = embedding as number[];
  } else {
    throw new Error('embedChunk: Unexpected embedding format');
  }
  if (!Array.isArray(embedding) || embedding.some((x) => typeof x !== 'number' || isNaN(x))) {
    throw new Error('embedChunk: Invalid embedding format after conversion');
  }
  return embedding;
}

async function processFile(file: string, collection: any) {
  const obj = loadArtObject(file);
  if (!obj) return;
  const text = buildChunkText(obj);
  const chunks = chunkTextTiktoken(text);
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (typeof chunk.text !== 'string') {
      const t: unknown = chunk.text;
      console.error(`[processFile] chunk.text is not a string for ${file} chunk ${i}:`, {
        type: typeof chunk.text,
        constructor: t && (t as any).constructor ? (t as any).constructor.name : undefined,
        preview: String(chunk.text).slice(0, 100),
      });
      continue;
    }
    try {
      const embedding = await embedChunk(chunk.text);
      if (!Array.isArray(embedding) || embedding.some((x) => typeof x !== 'number' || isNaN(x))) {
        throw new Error('Invalid embedding: ' + JSON.stringify(embedding));
      }
      console.log('Saving embedding:', {
        id: obj.id,
        chunkIndex: i,
        embedding: embedding.slice(0, 5),
        text: chunk.text.slice(0, 100),
      });
      await collection.add({
        ids: [`${obj.id}_${i}`],
        embeddings: [embedding],
        documents: [chunk.text],
        metadatas: [
          {
            objectId: obj.id,
            chunkIndex: i,
            start: chunk.start,
            end: chunk.end,
          },
        ],
      });
      console.log(`Saved: ${obj.id} chunk ${i + 1}/${chunks.length}`);
    } catch (e) {
      console.error(`Error embedding/saving ${obj.id} chunk ${i + 1}:`, e);
    }
  }
}

async function mainAsync() {
  const files = getAllJsonFiles(OBJECTS_DIR);
  console.log('Found', files.length, 'files');
  const collection = await getOrCreateCollection();
  for (const file of files) {
    await processFile(file, collection);
  }
}

mainAsync().catch(console.error);
