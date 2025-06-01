"""
🚀 FastAPI RAG Service - Versión Modular Mínima
Solo usando funciones que existen en database.py
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

# Import solo la función que sabemos que existe
from database import create_document
from processors.base_processor import ProcessorRegistry
from processors.excel_processor import ExcelProcessor

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = FastAPI(
    title="🤖 RAG Service - Modular Architecture",
    description="Sistema RAG empresarial con arquitectura modular",
    version="7.0.0-modular-minimal"
)

# CORS configuración
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

# ENDPOINTS

@app.get("/")
async def root():
    """Endpoint de información del servicio"""
    processor_info = processor_registry.get_processor_info()
    
    return {
        "message": "🤖 RAG Service - Modular Architecture (Minimal)",
        "status": "active",
        "environment": "production",
        "database": "Supabase",
        "ai": "OpenAI GPT-3.5",
        "architecture": "modular",
        "version": "7.0.0-modular-minimal",
        **processor_info
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    user_id: str = Query(...)
):
    """
    🚀 Endpoint modular para upload de documentos
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
        
        # Análisis avanzado
        advanced_analysis = await processor.generate_charts(file_content, file.filename)
        
        # Guardar con análisis enriquecido
        document_id = str(uuid.uuid4())
        
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
    💬 Endpoint de chat básico (sin búsqueda por ahora)
    """
    try:
        logger.info(f"💬 Chat request from user {request.user_id}: {request.message}")
        
        # Por ahora, respuesta simple hasta que tengamos búsqueda funcionando
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "Eres un asistente inteligente. El usuario puede subir documentos Excel para análisis. Por ahora, responde que la funcionalidad de búsqueda en documentos se está configurando."
                },
                {
                    "role": "user", 
                    "content": request.message
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        logger.info(f"✅ Chat response generated for user {request.user_id}")
        
        return {
            "response": response.choices[0].message.content + "\n\nNota: La búsqueda en documentos subidos se está configurando. Por ahora puedes subir archivos Excel para análisis usando el endpoint /analyze.",
            "sources": [],
            "document_count": 0,
            "status": "basic_mode"
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
    """🔧 Información de procesadores disponibles"""
    return processor_registry.get_processor_info()

# FUNCIÓN DE UTILIDAD

def save_document_to_supabase(document_id: str, user_id: str, filename: str, content: str, file_size: int):
    """Guardar documento usando database.py"""
    try:
        logger.info(f"📤 Saving document: {filename}")
        
        result = create_document(
            name=filename,
            content=content,
            size=file_size,
            user_id=user_id,
            metadata={'document_id': document_id}
        )
        
        logger.info(f"✅ Document saved: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error saving document: {e}")
        raise

# Inicialización
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting RAG Service with Modular Architecture (Minimal)")
    logger.info(f"📊 Processors: {[p.processor_name for p in processor_registry.processors]}")
    logger.info(f"📁 Extensions: {processor_registry.list_supported_extensions()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)