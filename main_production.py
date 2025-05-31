# main_production.py - Versión optimizada para Railway con Supabase + UPLOAD
import os
import ssl
import socket
import urllib3
from urllib3.util.ssl_ import create_urllib3_context
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import json
import re
import uuid
import PyPDF2
import io

# Configuración SSL para Pinecone
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ctx = create_urllib3_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
ctx.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
ssl._create_default_https_context = lambda: ctx
socket.setdefaulttimeout(30)

# Variables de entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("🔍 DEBUG: Starting main_production.py with Supabase integration")

# IMPORTAR CONFIGURACIÓN DE DATABASE.PY (CON SUPABASE HARDCODEADO)
try:
    from database import (
        get_db_cursor, 
        get_db_connection, 
        get_documents, 
        get_document_content,
        create_conversation,
        DATABASE_CONFIG
    )
    print(f"✅ SUCCESS: Imported database functions")
    print(f"🔍 DEBUG: Using database host: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"❌ ERROR: Failed to import database: {e}")
    DATABASE_AVAILABLE = False

# Configurar Pinecone
USE_PINECONE = PINECONE_API_KEY is not None
if USE_PINECONE:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        INDEX_NAME = "rag-cff-2048"
        pinecone_index = pc.Index(INDEX_NAME)
        print("✅ Pinecone configurado para producción")
    except Exception as e:
        print(f"⚠️ Error Pinecone: {e}")
        USE_PINECONE = False
        pinecone_index = None
else:
    pinecone_index = None

# Configurar OpenAI
USE_OPENAI = OPENAI_API_KEY is not None and OPENAI_API_KEY.startswith("sk-")
if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI configurado para producción")

# Crear app FastAPI
app = FastAPI(
    title="🤖 RAG Service Production",
    description="Servicio RAG híbrido en producción con Railway + Supabase",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS para permitir conexiones desde frontend
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos
class ChatRequest(BaseModel):
    question: str
    user_id: str = "119f7084-be9e-416f-81d6-3ffeadb062d5"  # ✅ UUID VÁLIDO

class ChatResponse(BaseModel):
    id: str
    question: str
    answer: str
    sources: List[dict]
    timestamp: str
    user_id: str
    ai_model: str

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    message: str
    file_size: int
    file_type: str

# 🆕 FUNCIONES PARA PROCESAMIENTO DE ARCHIVOS
def extract_text_from_pdf(content: bytes) -> str:
    """Extraer texto de PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return ""

def extract_text_from_file(content: bytes, content_type: str, filename: str) -> str:
    """Extraer texto según tipo de archivo"""
    try:
        if content_type == "application/pdf":
            return extract_text_from_pdf(content)
        elif content_type == "text/plain":
            return content.decode('utf-8', errors='ignore')
        elif filename.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        else:
            print(f"⚠️ Unsupported file type: {content_type}")
            return f"Archivo {filename} - contenido no procesable automáticamente"
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return f"Error procesando archivo {filename}"

def save_document_to_supabase(document_id: str, user_id: str, filename: str, content: str, file_size: int):
    """Guardar documento usando database.py - VERSIÓN CORREGIDA"""
    try:
        print(f"📤 DEBUG: Saving to database using database.py functions")
        print(f"📤 DEBUG: Filename: {filename}")
        print(f"📤 DEBUG: Content length: {len(content)} chars")
        
        # USAR LA FUNCIÓN DE database.py QUE YA FUNCIONA
        from database import create_document
        
        result = create_document(
            name=filename,
            content=content,
            size=file_size,
            user_id=user_id,
            metadata={'document_id': document_id}
        )
        
        print(f"✅ SUCCESS: Document saved via database.py: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error saving via database.py: {e}")
        raise

# Sinónimos y funciones de búsqueda
SYNONYMS = {
    'vacaciones': ['vacaciones', 'vacación', 'descanso', 'días libres', 'ausencia', 'permiso', 'tiempo libre'],
    'política': ['política', 'políticas', 'norma', 'normas', 'regla', 'reglas', 'procedimiento'],
    'solicitar': ['solicitar', 'pedir', 'requerir', 'tramitar', 'gestionar', 'obtener'],
    'trabajo': ['trabajo', 'laboral', 'empleo', 'empresa', 'oficina'],
    'horario': ['horario', 'horarios', 'tiempo', 'horas', 'jornada'],
    'manual': ['manual', 'guía', 'instructivo', 'documentación'],
    'sistema': ['sistema', 'plataforma', 'herramienta', 'aplicación']
}

def expand_query_terms(question: str) -> set:
    """Expandir términos de búsqueda"""
    question_lower = question.lower()
    expanded_terms = set()
    
    original_words = re.findall(r'\b\w+\b', question_lower)
    expanded_terms.update(original_words)
    
    for word in original_words:
        for key, synonyms in SYNONYMS.items():
            if word in synonyms:
                expanded_terms.update(synonyms)
    
    stop_words = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 
        'por', 'son', 'con', 'para', 'como', 'las', 'del', 'los', 'una', 'mas', 'pero', 'sus', 'muy',
        'qué', 'cómo', 'cuál', 'dónde', 'cuándo', 'quién'
    }
    
    return expanded_terms - stop_words

def calculate_relevance(content: str, doc_name: str, question_terms: set) -> float:
    """Calcular relevancia"""
    content_lower = content.lower()
    doc_name_lower = doc_name.lower()
    
    content_words = set(re.findall(r'\b\w+\b', content_lower))
    exact_matches = len(question_terms.intersection(content_words))
    content_relevance = exact_matches / max(len(question_terms), 1) if question_terms else 0
    
    doc_words = set(re.findall(r'\b\w+\b', doc_name_lower.replace('_', ' ').replace('.', ' ')))
    title_matches = len(question_terms.intersection(doc_words))
    title_relevance = (title_matches / max(len(question_terms), 1)) * 0.3 if question_terms else 0
    
    frequency_bonus = 0
    for term in question_terms:
        frequency_bonus += content_lower.count(term) * 0.05
    
    document_type_bonus = 0
    if 'política' in doc_name_lower:
        document_type_bonus += 0.2
    if 'manual' in doc_name_lower:
        document_type_bonus += 0.15
    
    total_relevance = content_relevance + title_relevance + min(frequency_bonus, 0.3) + document_type_bonus
    return min(total_relevance, 1.0)

def production_search(question: str, user_id: str = None) -> List[dict]:
    """Búsqueda optimizada para producción usando Supabase"""
    try:
        print(f"🔍 DEBUG: Searching for question: {question}")
        print(f"🔍 DEBUG: Using user_id: {user_id}")
        
        if not DATABASE_AVAILABLE:
            print("❌ ERROR: Database not available")
            return []
            
        question_terms = expand_query_terms(question)
        print(f"🔍 DEBUG: Expanded terms: {question_terms}")
        
        # USAR FUNCIÓN DE database.py (CON SUPABASE)
        docs_data = get_documents(user_id)
        print(f"🔍 DEBUG: Found {len(docs_data)} documents from Supabase")
        
        if not docs_data:
            print("⚠️ WARNING: No documents found in Supabase")
            return []
        
        processed_docs = [doc for doc in docs_data if doc.get('status') == 'processed']
        print(f"🔍 DEBUG: {len(processed_docs)} processed documents")
        
        relevant_docs = []
        for doc in processed_docs:
            try:
                # USAR FUNCIÓN DE database.py (CON SUPABASE)
                doc_content = get_document_content(doc['id'], user_id)
                if not doc_content:
                    continue
                    
                content = doc_content['content']
                relevance = calculate_relevance(content, doc['name'], question_terms)
                
                if relevance > 0.01:
                    # Extraer mejor fragmento
                    sentences = content.split('.')
                    best_excerpt = ""
                    best_score = 0
                    
                    for sentence in sentences[:20]:
                        sentence_lower = sentence.lower()
                        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
                        matches = len(question_terms.intersection(sentence_words))
                        
                        if matches > best_score:
                            best_score = matches
                            best_excerpt = sentence.strip()
                    
                    if not best_excerpt and sentences:
                        best_excerpt = sentences[0].strip()
                    
                    relevant_docs.append({
                        'document_id': doc['id'],
                        'document_name': doc['name'],
                        'content': content,
                        'excerpt': best_excerpt[:300] + "..." if len(best_excerpt) > 300 else best_excerpt,
                        'relevance': relevance
                    })
                    
            except Exception as e:
                print(f"❌ Error procesando documento: {e}")
                continue
        
        relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
        print(f"✅ SUCCESS: Found {len(relevant_docs)} relevant documents")
        return relevant_docs[:3]
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return []

def generate_production_answer(question: str, relevant_docs: List[dict]) -> str:
    """Generar respuesta para producción"""
    if not relevant_docs:
        return "Lo siento, no encontré información relevante en los documentos disponibles."
    
    if USE_OPENAI:
        try:
            context = "\n\n".join([
                f"DOCUMENTO: {doc['document_name']}\nCONTENIDO: {doc['excerpt']}"
                for doc in relevant_docs[:2]
            ])
            
            prompt = f"""Basándote en estos documentos empresariales, responde la pregunta de manera precisa y detallada:

{context}

PREGUNTA: {question}

Responde en español, cita las fuentes y sé específico."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en documentos empresariales."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error OpenAI: {e}")
    
    # Respuesta local
    best_doc = relevant_docs[0]
    return f"Según el documento '{best_doc['document_name']}': {best_doc['excerpt']}"

# Rutas de producción
@app.get("/")
async def root():
    return {
        "message": "🤖 RAG Service Production - Railway Deployment",
        "status": "active",
        "environment": "production",
        "database": f"Supabase ({DATABASE_CONFIG['host']})" if DATABASE_AVAILABLE else "Disconnected",
        "vector_search": "Pinecone" if USE_PINECONE else "Disabled",
        "ai": "OpenAI GPT-3.5" if USE_OPENAI else "Local",
        "version": "6.1.0-railway-supabase-upload"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if DATABASE_AVAILABLE else "disconnected",
            "pinecone": "enabled" if USE_PINECONE else "disabled",
            "openai": "enabled" if USE_OPENAI else "enabled"
        }
    }

# 🆕 NUEVO ENDPOINT DE UPLOAD
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "119f7084-be9e-416f-81d6-3ffeadb062d5"
):
    """Subir y procesar documento"""
    try:
        print(f"📤 DEBUG: Uploading file: {file.filename}")
        print(f"📤 DEBUG: Content type: {file.content_type}")
        print(f"📤 DEBUG: User ID: {user_id}")
        
        # Validar archivo
        allowed_types = [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        
        if file.content_type not in allowed_types and not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=400, 
                detail="Tipo de archivo no soportado. Solo PDF, TXT y DOCX."
            )
        
        # Leer contenido
        content = await file.read()
        file_size = len(content)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=400, detail="Archivo demasiado grande. Máximo 100MB.")
        
        # Extraer texto
        text_content = extract_text_from_file(content, file.content_type, file.filename)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del archivo.")
        
        # Generar ID único
        document_id = str(uuid.uuid4())
        
        # Guardar en Supabase
        save_document_to_supabase(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename,
            content=text_content,
            file_size=file_size
        )
        
        print(f"✅ SUCCESS: Document uploaded: {document_id}")
        
        return UploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            message="Documento subido y procesado exitosamente",
            file_size=file_size,
            file_type=file.content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.post("/chat")
async def production_chat(request: ChatRequest):
    """Chat endpoint para producción"""
    try:
        print(f"🔍 DEBUG: Chat request: {request.question}")
        print(f"🔍 DEBUG: Chat user_id: {request.user_id}")
        
        relevant_docs = production_search(request.question, request.user_id)
        answer = generate_production_answer(request.question, relevant_docs)
        
        sources = []
        for doc in relevant_docs:
            sources.append({
                'document': doc['document_name'],
                'excerpt': doc['excerpt'],
                'relevance': round(doc['relevance'], 3)
            })
        
        # USAR FUNCIÓN DE database.py (CON SUPABASE)
        if DATABASE_AVAILABLE:
            conv_result = create_conversation(
                question=request.question,
                answer=answer,
                sources=json.dumps(sources),
                user_id=request.user_id,
                ai_model='production-rag-railway-supabase'
            )
        else:
            conv_result = {"id": "error", "timestamp": datetime.now()}
        
        return ChatResponse(
            id=conv_result['id'],
            question=request.question,
            answer=answer,
            sources=sources,
            timestamp=conv_result['timestamp'].isoformat() if hasattr(conv_result['timestamp'], 'isoformat') else str(conv_result['timestamp']),
            user_id=request.user_id,
            ai_model='production-rag-railway-supabase'
        )
        
    except Exception as e:
        print(f"❌ ERROR in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/documents")
async def get_production_documents(user_id: str = None):
    """Obtener documentos en producción desde Supabase"""
    try:
        print(f"🔍 DEBUG: Getting documents for user: {user_id}")
        
        if not DATABASE_AVAILABLE:
            print("❌ ERROR: Database not available")
            return []
            
        # USAR FUNCIÓN DE database.py (CON SUPABASE)
        docs = get_documents(user_id)
        print(f"✅ SUCCESS: Retrieved {len(docs)} documents from Supabase")
        
        return [
            {
                "id": doc['id'],
                "name": doc['name'],
                "size": f"{doc['size'] / (1024*1024):.1f} MB" if isinstance(doc['size'], int) else str(doc['size']),
                "status": doc['status'],
                "upload_date": doc['upload_date'].isoformat() if hasattr(doc['upload_date'], 'isoformat') else str(doc['upload_date'])
            }
            for doc in docs
        ]
    except Exception as e:
        print(f"❌ ERROR getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)