import { HfInference } from '@huggingface/inference';
import { ChromaClient, IncludeEnum } from 'chromadb';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { pipeline } from '@xenova/transformers';

dotenv.config();

const hf = new HfInference(process.env.HF_API_TOKEN);
const chroma = new ChromaClient();
const COLLECTION_NAME = 'artsmia_chunks';

const HF_API_TOKEN = process.env.HF_API_TOKEN;
const EMBED_MODEL = 'BAAI/bge-small-en-v1.5';
const GEN_MODEL = 'mistralai/Mixtral-8x7B-Instruct-v0.1';

let extractor: any;
async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractor;
}

async function embedText(text: string): Promise<number[]> {
  const extractor = await getExtractor();
  const output = await extractor(text);
  return output.data[0];
}

async function getCollection() {
  return chroma.getCollection({
    name: COLLECTION_NAME,
    embeddingFunction: {} as unknown as any,
  });
}

function buildPrompt(chunks: { text: string }[], question: string): string {
  const context = chunks.map((c, i) => `Source [${i + 1}]:\n${c.text}`).join('\n\n');
  return `You are a helpful assistant for a museum collection. Use the provided sources to answer the question. If the answer is not in the sources, say you don't know.\n\n${context}\n\nQuestion: ${question}\nAnswer:`;
}

export async function answerQuestion(
  question: string
): Promise<{ answer: string; sources: { objectId: string; chunkIndex: number; text: string }[] }> {
  console.log('Embedding question (HF)...');
  const embedding = await embedText(question);
  const collection = await getCollection();
  console.log('Searching for relevant chunks...');
  const results = await collection.query({
    queryEmbeddings: [embedding],
    nResults: 5,
    include: [IncludeEnum.Documents, IncludeEnum.Metadatas],
  });
  const docs = results.documents?.[0] as string[] | undefined;
  const metas = results.metadatas?.[0] as any[] | undefined;
  const sources =
    docs && metas
      ? docs.map((text, i) => ({
          text,
          objectId: String(metas[i].objectId),
          chunkIndex: Number(metas[i].chunkIndex),
        }))
      : [];
  console.log('Building prompt and querying HF LLM...');
  const prompt = buildPrompt(sources, question);
  const genRes = await hf.textGeneration({
    model: GEN_MODEL,
    inputs: prompt,
    parameters: { max_new_tokens: 512, temperature: 0.2, return_full_text: false },
  });
  const answer =
    typeof genRes === 'object' && 'generated_text' in genRes ? genRes.generated_text : '';
  console.log('Answer generated.');
  return { answer, sources };
}
