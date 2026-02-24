# Voyage Toolbox

A comprehensive FastAPI wrapper for Voyage.ai, providing easy-to-use REST API endpoints for text embeddings, multimodal image embeddings, document reranking, and similarity calculations.

## Features

### Text Embeddings
- Single text embedding generation
- Batch text embedding generation (up to 1,000 texts)
- Text-to-text similarity calculation
- Support for multiple models:
  - `voyage-4-large`
  - `voyage-4` (default)
  - `voyage-4-lite`
  - `voyage-code-3`

### Multimodal Image Embeddings
- Image embedding generation (upload or URL)
- Text embedding using multimodal model
- Image-to-image similarity
- Image-to-text similarity

### Document Reranking
- Rerank documents by relevance to a query
- Support for multiple reranker models:
  - `rerank-2.5` (default) - Optimized for quality
  - `rerank-2.5-lite` - Optimized for latency and quality
- Top-K filtering support
- Up to 1,000 documents per request
- 32,000 token context length

### Similarity Metrics
All similarity endpoints return three metrics:
- Euclidean distance
- Dot product
- Cosine similarity

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Voyage.ai API key:
```bash
export VOYAGE_API_KEY="your-api-key-here"
```

## Running the Service

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### General

- `GET /` - Root endpoint with API information
- `GET /models` - List all available models

### Text Embedding Endpoints

#### Single Text Embedding
`POST /text/embed`

Generate embedding for a single text.

**Request Body:**
```json
{
  "text": "Your text here",
  "model": "voyage-4",
  "input_type": null,
  "truncation": true,
  "output_dimension": null
}
```

**Parameters:**
- `text` (required): The input text to embed
- `model` (optional): Model to use (default: "voyage-4")
- `input_type` (optional): "query", "document", or null
- `truncation` (optional): Whether to truncate if too long (default: true)
- `output_dimension` (optional): 256, 512, 1024, or 2048

#### Batch Text Embedding
`POST /text/embed/batch`

Generate embeddings for multiple texts (max 1,000).

**Request Body:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "voyage-4",
  "input_type": null,
  "truncation": true,
  "output_dimension": null
}
```

#### Text-to-Text Similarity
`POST /text/similarity`

Calculate similarity between two texts using a single model.

**Request Body:**
```json
{
  "text1": "First text",
  "text2": "Second text",
  "model": "voyage-4",
  "input_type": "query",
  "truncation": true,
  "output_dimension": 1024
}
```

#### Text-to-Text Similarity (All Models)
`POST /text/similarity/all-models`

Calculate similarity between two texts using ALL available text embedding models at once.

**Request Body:**
```json
{
  "text1": "First text",
  "text2": "Second text",
  "input_type": "query",
  "truncation": true,
  "output_dimension": 1024
}
```

**Response:**
```json
{
  "results": [
    {
      "model": "voyage-3-large",
      "euclidean": 0.45,
      "dotProduct": 0.87,
      "cosine": 0.93
    },
    {
      "model": "voyage-4",
      "euclidean": 0.42,
      "dotProduct": 0.89,
      "cosine": 0.94
    }
  ]
}
```

### Image Embedding Endpoints

#### Upload Image Vector
`POST /image/upload/vector`

Generate embedding for an uploaded image.

**Form Data:**
- `image`: Image file

#### URL Image Vector
`POST /image/url/vector`

Generate embedding for an image from URL.

**Query Parameter:**
- `image_url`: URL of the image

#### Multimodal Text Vector
`POST /image/text/vector`

Generate text embedding using the multimodal model.

**Query Parameter:**
- `text`: Text to embed

### Similarity Endpoints

#### Upload Image-Image Similarity
`POST /image/upload/similarity`

Calculate similarity between two uploaded images.

**Form Data:**
- `image1`: First image file
- `image2`: Second image file

#### URL Image-Image Similarity
`POST /image/url/similarity`

Calculate similarity between two images from URLs.

**Query Parameters:**
- `image_url1`: URL of first image
- `image_url2`: URL of second image

#### Upload Image-Text Similarity
`POST /image/upload/text/similarity`

Calculate similarity between an uploaded image and text.

**Form Data:**
- `image`: Image file

**Query Parameter:**
- `text`: Text to compare

#### URL Image-Text Similarity
`POST /image/url/text/similarity`

Calculate similarity between an image from URL and text.

**Query Parameters:**
- `image_url`: URL of the image
- `text`: Text to compare

### Reranker Endpoints

#### Rerank Documents
`POST /rerank`

Rerank documents based on their relevance to a query.

**Request Body:**
```json
{
  "query": "When is Apple's conference call scheduled?",
  "documents": [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
    "Apple's conference call is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT.",
    "Photosynthesis in plants converts light energy into glucose."
  ],
  "model": "rerank-2.5",
  "top_k": 3,
  "truncation": true
}
```

**Parameters:**
- `query` (required): The search query string
- `documents` (required): List of documents to rerank (max 1,000)
- `model` (optional): Reranker model to use (default: "rerank-2.5")
  - `rerank-2.5`: Optimized for quality
  - `rerank-2.5-lite`: Optimized for latency and quality
- `top_k` (optional): Return only top K most relevant documents
- `truncation` (optional): Whether to truncate inputs if too long (default: true)

**Response:**
```json
{
  "results": [
    {
      "index": 1,
      "document": "Apple's conference call is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT.",
      "relevance_score": 0.98
    },
    {
      "index": 0,
      "document": "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
      "relevance_score": 0.12
    }
  ],
  "total_tokens": 1250,
  "model": "rerank-2.5"
}
```

## Example Usage

### Using curl

**Single text embedding:**
```bash
curl -X POST "http://localhost:8000/text/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
    "model": "voyage-4"
  }'
```

**Batch text embedding:**
```bash
curl -X POST "http://localhost:8000/text/embed/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First document",
      "Second document",
      "Third document"
    ],
    "model": "voyage-4",
    "input_type": "document"
  }'
```

**Text similarity:**
```bash
curl -X POST "http://localhost:8000/text/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "I like cats",
    "text2": "I also like dogs",
    "model": "voyage-4"
  }'
```

**Image upload:**
```bash
curl -X POST "http://localhost:8000/image/upload/vector" \
  -F "image=@path/to/your/image.jpg"
```

**Rerank documents:**
```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When is Apple'\''s conference call?",
    "documents": [
      "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
      "Apple'\''s conference call is scheduled for Thursday, November 2, 2023.",
      "Photosynthesis in plants converts light energy into glucose."
    ],
    "model": "rerank-2.5",
    "top_k": 2
  }'
```

### Using Python

```python
import requests

# Text embedding
response = requests.post(
    "http://localhost:8000/text/embed",
    json={
        "text": "Your text here",
        "model": "voyage-4"
    }
)
embedding = response.json()["embedding"]

# Batch text embedding
response = requests.post(
    "http://localhost:8000/text/embed/batch",
    json={
        "texts": ["Text 1", "Text 2", "Text 3"],
        "model": "voyage-4",
        "input_type": "document"
    }
)
embeddings = response.json()["embeddings"]

# Text similarity
response = requests.post(
    "http://localhost:8000/text/similarity",
    json={
        "text1": "I like programming",
        "text2": "I enjoy coding",
        "model": "voyage-4"
    }
)
similarity = response.json()
print(f"Cosine similarity: {similarity['cosine']}")

# Image embedding
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/image/upload/vector",
        files={"image": f}
    )
image_embedding = response.json()["embedding"]

# Rerank documents
response = requests.post(
    "http://localhost:8000/rerank",
    json={
        "query": "When is Apple's conference call?",
        "documents": [
            "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
            "Apple's conference call is scheduled for Thursday, November 2, 2023.",
            "Photosynthesis in plants converts light energy into glucose."
        ],
        "model": "rerank-2.5",
        "top_k": 2
    }
)
results = response.json()
for result in results["results"]:
    print(f"Score: {result['relevance_score']:.3f} - {result['document']}")
```

## Model Selection

### Text Embedding Models

Choose based on your needs:

- **voyage-4**: Best overall performance (default)
- **voyage-4-lite**: Faster and more cost-effective
- **voyage-3-large**: High quality embeddings
- **voyage-code-3**: Optimized for code and technical content

### Multimodal Models

- **voyage-multimodal-3.5**: For image and text embeddings, cross-modal similarity

### Reranker Models

- **rerank-2.5**: Generalist reranker optimized for quality (default)
- **rerank-2.5-lite**: Optimized for both latency and quality

## License

MIT License

## Contact

Pat Wendorf - pat.wendorf@mongodb.com
