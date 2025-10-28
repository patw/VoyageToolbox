from fastapi import FastAPI, UploadFile, Query
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel
import numpy
from scipy.spatial.distance import euclidean
import voyageai
from io import BytesIO
import requests

app = FastAPI(
    title="Voyage.ai Toolbox - Embeddings and Rerankers",
    description="API toolbox for Voyage.ai embeddings and rerankers including text embeddings (voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3) and multimodal image embeddings",
    version="1.0",
    contact={
        "name": "Pat Wendorf",
        "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/license/mit/",
    }
)

# Initialize VoyageAI client
vo = voyageai.Client()

# Supported text embedding models
TEXT_MODELS = ["voyage-3-large", "voyage-3.5", "voyage-3.5-lite", "voyage-code-3"]

# Supported reranker models
RERANKER_MODELS = ["rerank-2.5", "rerank-2.5-lite"]

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str
    model: str = "voyage-3.5"
    input_type: Optional[str] = "document"
    truncation: Optional[bool] = True
    output_dimension: Optional[int] = 1024

class BatchTextInput(BaseModel):
    texts: List[str]
    model: str = "voyage-3.5"
    input_type: Optional[str] = "document"
    truncation: Optional[bool] = True
    output_dimension: Optional[int] = 1024

class TextSimilarityInput(BaseModel):
    text1: str
    text2: str
    model: str = "voyage-3.5"
    input_type: Optional[str] = "document"
    truncation: Optional[bool] = True
    output_dimension: Optional[int] = 1024

class MultiModelTextSimilarityInput(BaseModel):
    text1: str
    text2: str
    input_type: Optional[str] = "document"
    truncation: Optional[bool] = True
    output_dimension: Optional[int] = 1024

class SimilarityResponse(BaseModel):
    euclidean: float
    dotProduct: float
    cosine: float

class RerankInput(BaseModel):
    query: str
    documents: List[str]
    model: str = "rerank-2.5"
    top_k: Optional[int] = 5
    truncation: Optional[bool] = True

# Utility functions
def similarity(v1, v2):
    """Calculate similarity metrics between two vectors"""
    vector1 = numpy.array(v1)
    vector2 = numpy.array(v2)

    # Compute Euclidean distance
    euclidean_distance = euclidean(vector1, vector2)

    # Compute dot product
    dot_product = numpy.dot(vector1, vector2)

    # Compute cosine similarity
    cosine_similarity = numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    return {"euclidean": float(euclidean_distance), "dotProduct": float(dot_product), "cosine": float(cosine_similarity)}

def get_voyage_text_embedding(text: str, model: str = "voyage-3.5", input_type: Optional[str] = None,
                               truncation: bool = True, output_dimension: Optional[int] = None):
    """Get text embedding using Voyage.ai text embedding models"""
    result = vo.embed(
        texts=[text],
        model=model,
        input_type=input_type,
        truncation=truncation,
        output_dimension=output_dimension
    )
    return result.embeddings[0]

def get_voyage_batch_text_embedding(texts: List[str], model: str = "voyage-3.5", input_type: Optional[str] = None,
                                    truncation: bool = True, output_dimension: Optional[int] = None):
    """Get batch text embeddings using Voyage.ai text embedding models"""
    result = vo.embed(
        texts=texts,
        model=model,
        input_type=input_type,
        truncation=truncation,
        output_dimension=output_dimension
    )
    return result.embeddings

def get_voyage_image_embedding(image):
    """Get image embedding using Voyage.ai multimodal model"""
    result = vo.multimodal_embed([[image]], model="voyage-multimodal-3")
    return result.embeddings[0]

def get_voyage_multimodal_text_embedding(text):
    """Get text embedding using Voyage.ai multimodal model"""
    result = vo.multimodal_embed([[text]], model="voyage-multimodal-3")
    return result.embeddings[0]

def get_voyage_rerank(query: str, documents: List[str], model: str = "rerank-2.5",
                      top_k: Optional[int] = None, truncation: bool = True):
    """Rerank documents using Voyage.ai reranker models"""
    result = vo.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=top_k,
        truncation=truncation
    )
    return result

# ========== TEXT EMBEDDING ENDPOINTS ==========

@app.get("/models")
async def list_models():
    """List all available models"""
    return {
        "text_models": TEXT_MODELS,
        "multimodal_models": ["voyage-multimodal-3"],
        "reranker_models": RERANKER_MODELS
    }

@app.post("/text/embed")
async def embed_single_text(input_data: TextInput):
    """
    Generate embedding vector for a single text using specified model.

    - **text**: The input text to embed
    - **model**: Model to use (default: voyage-3.5)
    - **input_type**: Optional type hint for the text (None, "query", or "document")
    - **truncation**: Whether to truncate text if too long (default: True)
    - **output_dimension**: Optional output dimension (256, 512, 1024, 2048)
    """
    embedding = get_voyage_text_embedding(
        text=input_data.text,
        model=input_data.model,
        input_type=input_data.input_type,
        truncation=input_data.truncation,
        output_dimension=input_data.output_dimension
    )
    return {"embedding": embedding, "model": input_data.model}

@app.post("/text/embed/batch")
async def embed_batch_text(input_data: BatchTextInput):
    """
    Generate embedding vectors for multiple texts (batch processing).

    - **texts**: List of input texts to embed (max 1,000)
    - **model**: Model to use (default: voyage-3.5)
    - **input_type**: Optional type hint for the texts (None, "query", or "document")
    - **truncation**: Whether to truncate text if too long (default: True)
    - **output_dimension**: Optional output dimension (256, 512, 1024, 2048)
    """
    embeddings = get_voyage_batch_text_embedding(
        texts=input_data.texts,
        model=input_data.model,
        input_type=input_data.input_type,
        truncation=input_data.truncation,
        output_dimension=input_data.output_dimension
    )
    return {"embeddings": embeddings, "count": len(embeddings), "model": input_data.model}

@app.post("/text/similarity")
async def text_to_text_similarity(input_data: TextSimilarityInput):
    """
    Calculate similarity between two texts using specified model.
    Returns Euclidean distance, dot product, and cosine similarity.

    - **text1**: First text
    - **text2**: Second text
    - **model**: Model to use (default: voyage-3.5)
    - **input_type**: Optional type hint for the texts (None, "query", or "document")
    - **truncation**: Whether to truncate text if too long (default: True)
    - **output_dimension**: Optional output dimension (256, 512, 1024, 2048)
    """
    v1 = get_voyage_text_embedding(
        text=input_data.text1,
        model=input_data.model,
        input_type=input_data.input_type,
        truncation=input_data.truncation,
        output_dimension=input_data.output_dimension
    )
    v2 = get_voyage_text_embedding(
        text=input_data.text2,
        model=input_data.model,
        input_type=input_data.input_type,
        truncation=input_data.truncation,
        output_dimension=input_data.output_dimension
    )
    return similarity(v1, v2)

@app.post("/text/similarity/all-models")
async def text_to_text_similarity_all_models(input_data: MultiModelTextSimilarityInput):
    """
    Calculate similarity between two texts using ALL available text embedding models.
    Returns an array with results for each model including Euclidean distance, dot product, and cosine similarity.

    - **text1**: First text
    - **text2**: Second text
    - **input_type**: Optional type hint for the texts (None, "query", or "document")
    - **truncation**: Whether to truncate text if too long (default: True)
    - **output_dimension**: Optional output dimension (256, 512, 1024, 2048)
    """
    results = []

    for model in TEXT_MODELS:
        v1 = get_voyage_text_embedding(
            text=input_data.text1,
            model=model,
            input_type=input_data.input_type,
            truncation=input_data.truncation,
            output_dimension=input_data.output_dimension
        )
        v2 = get_voyage_text_embedding(
            text=input_data.text2,
            model=model,
            input_type=input_data.input_type,
            truncation=input_data.truncation,
            output_dimension=input_data.output_dimension
        )
        scores = similarity(v1, v2)
        results.append({
            "model": model,
            **scores
        })

    return {"results": results}

# ========== MULTIMODAL IMAGE EMBEDDING ENDPOINTS ==========

@app.post("/image/upload/vector")
async def upload_image_vector(image: UploadFile):
    """
    Generate embedding vector for an uploaded image using voyage-multimodal-3.
    """
    img = Image.open(image.file)
    embedding = get_voyage_image_embedding(img)
    return {"embedding": embedding, "model": "voyage-multimodal-3"}

@app.post("/image/url/vector")
async def url_image_vector(image_url: str):
    """
    Generate embedding vector for an image from URL using voyage-multimodal-3.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    embedding = get_voyage_image_embedding(img)
    return {"embedding": embedding, "model": "voyage-multimodal-3"}

@app.post("/image/text/vector")
async def multimodal_text_vector(text: str):
    """
    Generate embedding vector for text using voyage-multimodal-3 model.
    """
    embedding = get_voyage_multimodal_text_embedding(text)
    return {"embedding": embedding, "model": "voyage-multimodal-3"}

# ========== IMAGE SIMILARITY ENDPOINTS ==========

@app.post("/image/upload/similarity")
async def upload_image_image_similarity(image1: UploadFile, image2: UploadFile):
    """
    Calculate similarity between two uploaded images.
    Returns Euclidean distance, dot product, and cosine similarity.
    """
    i1 = Image.open(image1.file)
    i2 = Image.open(image2.file)
    v1 = get_voyage_image_embedding(i1)
    v2 = get_voyage_image_embedding(i2)
    return similarity(v1, v2)

@app.post("/image/url/similarity")
async def url_image_image_similarity(image_url1: str, image_url2: str):
    """
    Calculate similarity between two images from URLs.
    Returns Euclidean distance, dot product, and cosine similarity.
    """
    r1 = requests.get(image_url1)
    r2 = requests.get(image_url2)
    i1 = Image.open(BytesIO(r1.content))
    i2 = Image.open(BytesIO(r2.content))
    v1 = get_voyage_image_embedding(i1)
    v2 = get_voyage_image_embedding(i2)
    return similarity(v1, v2)

# ========== IMAGE-TEXT SIMILARITY ENDPOINTS ==========

@app.post("/image/upload/text/similarity")
async def upload_image_text_similarity(image: UploadFile, text: str):
    """
    Calculate similarity between an uploaded image and text using voyage-multimodal-3.
    Returns Euclidean distance, dot product, and cosine similarity.
    """
    img = Image.open(image.file)
    v1 = get_voyage_image_embedding(img)
    v2 = get_voyage_multimodal_text_embedding(text)
    return similarity(v1, v2)

@app.post("/image/url/text/similarity")
async def url_image_text_similarity(image_url: str, text: str):
    """
    Calculate similarity between an image from URL and text using voyage-multimodal-3.
    Returns Euclidean distance, dot product, and cosine similarity.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    v1 = get_voyage_image_embedding(img)
    v2 = get_voyage_multimodal_text_embedding(text)
    return similarity(v1, v2)

# ========== RERANKER ENDPOINTS ==========

@app.post("/rerank")
async def rerank_documents(input_data: RerankInput):
    """
    Rerank documents based on their relevance to a query using Voyage.ai reranker models.

    - **query**: The search query string
    - **documents**: List of documents to rerank (max 1,000)
    - **model**: Reranker model to use (default: rerank-2.5)
    - **top_k**: Optional - return only the top K most relevant documents
    - **truncation**: Whether to truncate inputs if too long (default: True)
    """
    result = get_voyage_rerank(
        query=input_data.query,
        documents=input_data.documents,
        model=input_data.model,
        top_k=input_data.top_k,
        truncation=input_data.truncation
    )

    # Convert result to serializable format
    return {
        "results": [
            {
                "index": r.index,
                "document": r.document,
                "relevance_score": r.relevance_score
            }
            for r in result.results
        ],
        "total_tokens": result.total_tokens,
        "model": input_data.model
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to VoyageToolbox API",
        "docs": "/docs",
        "text_models": TEXT_MODELS,
        "multimodal_models": ["voyage-multimodal-3"],
        "reranker_models": RERANKER_MODELS
    }
