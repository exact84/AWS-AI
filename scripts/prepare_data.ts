import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { HfInference } from '@huggingface/inference';
import dotenv from 'dotenv';
import { encoding_for_model } from 'tiktoken';
import { ChromaClient } from 'chromadb';
import fetch from 'node-fetch';
import { pipeline } from '@xenova/transformers';

dotenv.config();

const hf = new HfInference(process.env.HF_API_TOKEN);
// console.log(hf);

const EMBED_MODELS = ['sentence-transformers/all-MiniLM-L6-v2'];
const EMBED_MODEL = 'BAAI/bge-small-en-v1.5';

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
const MODEL_NAME = 'gpt-3.5-turbo';

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
    await chroma.createCollection({ name: COLLECTION_NAME });
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
    if (!raw.trim()) return null; // пустой файл
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
    .join('\n');
}

function chunkTextTiktoken(
  text: string
): { text: string; start: number; end: number; tokens: number }[] {
  const enc = encoding_for_model(MODEL_NAME);
  const tokens = enc.encode(text);
  const chunks: any[] = [];
  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + CHUNK_SIZE, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    const chunkText = enc.decode(chunkTokens);
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
  const extractor = await getExtractor();
  const output = await extractor(text);
  return output.data[0];
}

async function processFile(file: string, collection: any) {
  const obj = loadArtObject(file);
  if (!obj) return;
  const text = buildChunkText(obj);
  const chunks = chunkTextTiktoken(text);
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    try {
      const embedding = await embedChunk(chunk.text);
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

// Для запуска асинхронного main
mainAsync().catch(console.error);
