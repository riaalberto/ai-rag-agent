"""
🚀 FastAPI RAG Service - Versión Modular y Escalable
Arquitectura limpia con procesadores modulares para diferentes tipos de archivos
"""

import os
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import logging

# Imports de módulos locales - CORREGIDO
from database import create_document, get_documents_by_user, search_similar_documents
from processors.base_processor import ProcessorRegistry
from processors.excel_processor import ExcelProcessor

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = FastAPI(
    title="🤖 RAG Service - Modular Architecture",
    description="Sistema RAG empresarial con arquitectura modular para múltiples tipos de archivos",
    version="7.0.0-modular"
)

# CORS configuración mejorada
cors_origins = os.getenv("CORS_ORIGINS", "*")
logger.info(f"🔍 DEBUG: CORS_ORIGINS value: {cors_origins}")

if cors_origins == "*":
    allowed_origins = ["*"]
    logger.info("🌐 CORS: Allowing all origins (*)")
else:
    allowed_origins = cors_origins.split(",")
    logger.info(f"🌐 CORS: Allowing specific origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configuración de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("❌ OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY environment variable is required")

logger.info("✅ OpenAI configurado para producción")

# Inicializar registro de procesadores
processor_registry = ProcessorRegistry()

# Registrar procesadores disponibles
processor_registry.register(ExcelProcessor())

logger.info(f"🏗️ Initialized with {len(processor_registry.processors)} processors")

# Modelos Pydantic
class ChatRequest(BaseModel):
    message: str
    user_id: str

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    message: str
    file_size: int
    processor_used: str

# FUNCIÓN DE BÚSQUEDA TEMPORAL (hasta que esté en database.py)
def search_documents(query: str, user_id: str):
    """
    Función temporal de búsqueda de documentos
    Busca documentos por user_id y filtra por relevancia básica
    """
    try:
        # Obtener todos los documentos del usuario
        all_docs = get_documents_by_user(user_id)
        
        if not all_docs:
            return []
        
        # Filtro básico por palabras clave
        query_words = query.lower().split()
        relevant_docs = []
        
        for doc in all_docs:
            content_lower = doc.get('content', '').lower()
            name_lower = doc.get('name', '').lower()
            
            # Calcular relevancia básica
            relevance_score = 0
            for word in query_words:
                if word in content_lower:
                    relevance_score += content_lower.count(word)
                if word in name_lower:
                    relevance_score += 5  # Bonus por estar en el nombre
            
            if relevance_score > 0:
                doc['relevance_score'] = relevance_score
                relevant_docs.append(doc)
        
        # Ordenar por relevancia
        relevant_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"🔍 Found {len(relevant_docs)} relevant documents for query: {query}")
        return relevant_docs[:5]  # Top 5 más relevantes
        
    except Exception as e:
        logger.error(f"❌ Error searching documents: {e}")
        return []

# ENDPOINTS

@app.get("/")
async def root():
    """Endpoint de información del servicio"""
    processor_info = processor_registry.get_processor_info()
    
    return {
        "message": "🤖 RAG Service - Modular Architecture",
        "status": "active",
        "environment": "production",
        "database": "Supabase (aws-0-us-east-2.pooler.supabase.com)",
        "ai": "OpenAI GPT-3.5",
        "architecture": "modular",
        "version": "7.0.0-modular",
        **processor_info
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    user_id: str = Query(...)
):
    """
    🚀 Endpoint modular para upload de documentos
    Automáticamente selecciona el procesador adecuado según el tipo de archivo
    """
    try:
        logger.info(f"📤 Upload request: {file.filename} by user {user_id}")
        
        # Encontrar procesador adecuado
        processor = processor_registry.get_processor(file.filename)
        if not processor:
            supported_extensions = processor_registry.list_supported_extensions()
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
                }
            )
        
        # Leer contenido del archivo
        file_content = await file.read()
        
        # Validar archivo
        is_valid, validation_message = processor.validate_file(file_content, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"File validation failed: {validation_message}"
                }
            )
        
        # Procesar archivo
        result = await processor.process_file(file_content, file.filename)
        
        if not result["success"]:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result["error"]
                }
            )
        
        # Guardar en base de datos
        document_id = str(uuid.uuid4())
        
        save_success = save_document_to_supabase(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename,
            content=result["extracted_text"],
            file_size=len(file_content)
        )
        
        if save_success:
            logger.info(f"✅ Document saved successfully: {document_id}")
            
            return UploadResponse(
                success=True,
                document_id=document_id,
                filename=file.filename,
                message="Documento procesado y guardado exitosamente",
                file_size=len(file_content),
                processor_used=processor.processor_name
            )
        else:
            raise Exception("Failed to save document to database")
            
    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg
            }
        )

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    user_id: str = Query(...)
):
    """
    🧠 Endpoint para análisis avanzado de documentos
    Disponible para procesadores que soporten análisis avanzado
    """
    try:
        logger.info(f"🧠 Analysis request: {file.filename}")
        
        # Encontrar procesador
        processor = processor_registry.get_processor(file.filename)
        if not processor:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Unsupported file type for analysis"
                }
            )
        
        # Verificar si el procesador soporta análisis avanzado
        if not hasattr(processor, 'generate_charts'):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"{processor.processor_name} does not support advanced analysis"
                }
            )
        
        # Leer y procesar archivo
        file_content = await file.read()
        
        # Procesar archivo básico
        basic_result = await processor.process_file(file_content, file.filename)
        
        if not basic_result["success"]:
            return JSONResponse(
                status_code=500,
                content=basic_result
            )
        
        # Análisis avanzado (ej: gráficas para Excel)
        advanced_analysis = await processor.generate_charts(file_content, file.filename)
        
        # Guardar con análisis enriquecido
        document_id = str(uuid.uuid4())
        
        # Crear contenido enriquecido
        enriched_content = basic_result["extracted_text"]
        if basic_result.get("analysis"):
            analysis_text = f"\n\nANÁLISIS AVANZADO:\n{basic_result['analysis']}"
            enriched_content += analysis_text
        
        save_success = save_document_to_supabase(
            document_id=document_id,
            user_id=user_id,
            filename=f"ANALYSIS_{file.filename}",
            content=enriched_content,
            file_size=len(file_content)
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "basic_analysis": basic_result.get("analysis"),
            "advanced_analysis": advanced_analysis,
            "processor_used": processor.processor_name,
            "message": "Análisis completo realizado exitosamente"
        }
        
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg
            }
        )

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """
    💬 Endpoint de chat inteligente
    Busca en documentos y responde usando OpenAI
    """
    try:
        logger.info(f"💬 Chat request from user {request.user_id}: {request.message}")
        
        # Buscar documentos relevantes
        documents = search_documents(request.message, request.user_id)
        
        if not documents:
            return {
                "response": "No encontré documentos relevantes para tu pregunta. ¿Podrías subir algunos documentos primero?",
                "sources": [],
                "document_count": 0
            }
        
        # Preparar contexto
        context = "\n\n".join([doc['content'] for doc in documents[:3]])
        
        # Detectar si es pregunta sobre análisis
        analysis_keywords = ['gráfica', 'análisis', 'estadística', 'insight', 'patrón', 'recomenda']
        is_analysis_question = any(keyword in request.message.lower() for keyword in analysis_keywords)
        
        # Configurar prompt según el tipo de pregunta
        if is_analysis_question:
            system_prompt = """Eres un asistente experto en análisis de datos empresariales.
            
Cuando respondas sobre archivos de Excel o análisis:
1. Proporciona insights claros y accionables
2. Menciona patrones y tendencias importantes
3. Sugiere análisis adicionales si es relevante
4. Usa un lenguaje profesional pero accesible
5. Si hay gráficas disponibles, menciónalas

Siempre cita las fuentes específicas de donde obtienes la información."""
        else:
            system_prompt = """Eres un asistente inteligente que ayuda a responder preguntas basándose en documentos empresariales.

Responde de manera clara y precisa, citando siempre las fuentes específicas.
Si la información no está en los documentos, menciona que necesitas más contexto."""
        
        # Llamada a OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"Contexto de documentos:\n{context}\n\nPregunta del usuario: {request.message}"
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        logger.info(f"✅ Chat response generated for user {request.user_id}")
        
        return {
            "response": response.choices[0].message.content,
            "sources": [doc['name'] for doc in documents[:3]],
            "document_count": len(documents),
            "analysis_mode": is_analysis_question
        }
        
    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg
            }
        )

@app.get("/processors")
async def get_processors_info():
    """
    🔧 Endpoint para obtener información de procesadores disponibles
    """
    return processor_registry.get_processor_info()

# FUNCIONES DE UTILIDAD (mantenidas del archivo original)

def save_document_to_supabase(document_id: str, user_id: str, filename: str, content: str, file_size: int):
    """Guardar documento usando database.py - VERSIÓN CORREGIDA"""
    try:
        logger.info(f"📤 DEBUG: Saving to database using database.py functions")
        logger.info(f"📤 DEBUG: Filename: {filename}")
        logger.info(f"📤 DEBUG: Content length: {len(content)} chars")
        
        # USAR LA FUNCIÓN DE database.py QUE YA FUNCIONA
        result = create_document(
            name=filename,
            content=content,
            size=file_size,
            user_id=user_id,
            metadata={'document_id': document_id}
        )
        
        logger.info(f"✅ SUCCESS: Document saved via database.py: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error saving via database.py: {e}")
        raise

# Inicialización del servidor
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting RAG Service with Modular Architecture")
    logger.info(f"📊 Processors available: {[p.processor_name for p in processor_registry.processors]}")
    logger.info(f"📁 Supported extensions: {processor_registry.list_supported_extensions()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)