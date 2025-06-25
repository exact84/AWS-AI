import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { answerQuestion } from './rag_pipeline.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.post('/api/ask', async (req, res) => {
  const { question } = req.body;
  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ error: 'Question is required and must be a non-empty string.' });
  }
  try {
    const result = await answerQuestion(question.trim());
    res.json(result);
  } catch (e) {
    console.error('Error in /api/ask:', e);
    res.status(500).json({ error: 'Internal server error.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
