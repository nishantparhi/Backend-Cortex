const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize OpenAI
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// In-memory vector store
let vectorStore = [];

// Helper: Calculate Cosine Similarity
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Helper: Load and Embed Knowledge Base
async function initializeKnowledgeBase() {
    console.log('Initializing Knowledge Base...');
    const kbPath = path.join(__dirname, 'knowledge-base', 'profile.md');

    try {
        const content = fs.readFileSync(kbPath, 'utf-8');
        // Split by headers (simple chunking)
        const sections = content.split(/^## /gm).filter(s => s.trim().length > 0);

        for (const section of sections) {
            const text = '## ' + section.trim(); // Re-add header for context

            // Generate embedding
            const embeddingResponse = await openai.embeddings.create({
                model: "text-embedding-ada-002",
                input: text,
            });

            const embedding = embeddingResponse.data[0].embedding;

            vectorStore.push({
                text,
                embedding
            });
        }
        console.log(`Knowledge Base loaded with ${vectorStore.length} chunks.`);
    } catch (error) {
        console.error('Error loading knowledge base:', error);
    }
}

// API Endpoint: Chat
app.post('/api/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    try {
        // 1. Embed User Query
        const embeddingResponse = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: message,
        });
        const queryEmbedding = embeddingResponse.data[0].embedding;

        // 2. Search Vector Store
        const scoredChunks = vectorStore.map(chunk => ({
            text: chunk.text,
            score: cosineSimilarity(queryEmbedding, chunk.embedding)
        }));

        // Sort by score and take top 3
        scoredChunks.sort((a, b) => b.score - a.score);
        const topChunks = scoredChunks.slice(0, 3);
        const context = topChunks.map(c => c.text).join('\n\n');

        // 3. Generate Response
        const systemPrompt = `
You are a personalized AI assistant for Nishant Parhi.
Your role is to answer questions about Nishant’s professional background, achievements, skills, and projects.
Only use the information provided from the knowledge base below.
Do NOT make up facts.
If the answer cannot be found in the provided context, clearly state that the information is not available.

Knowledge Base Context:
${context}
`;

        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: message }
            ],
            temperature: 0.7,
        });

        const reply = completion.choices[0].message.content;
        res.json({ reply });

    } catch (error) {
        console.error('Error processing chat:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Start Server
app.listen(PORT, async () => {
    console.log(`Server running on http://localhost:${PORT}`);
    // Only initialize if API key is present
    if (process.env.OPENAI_API_KEY) {
        await initializeKnowledgeBase();
    } else {
        console.warn('OPENAI_API_KEY not found. Knowledge base not loaded.');
    }
});
