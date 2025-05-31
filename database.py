"""
Configuración de base de datos PostgreSQL para AI-RAG-Agent
Maneja conexiones y operaciones de base de datos
"""

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import logging
from urllib.parse import urlparse

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURACIÓN DE BASE DE DATOS
# ============================================

def get_database_config():
    """Obtener configuración de base de datos - Hardcodeada para Supabase"""
    
    print("🔍 DEBUG: Using hardcoded Supabase configuration!")
    logger.info("Using hardcoded Supabase configuration")
    
    # Configuración directa de Supabase (evita problemas con variables de Railway)
    return {
        'host': 'aws-0-us-east-2.pooler.supabase.com',
        'database': 'postgres',
        'user': 'postgres.peeljvqscrkqmdbvfeag',
        'password': 'Monsemonsemo1$',
        'port': 6543,
        'sslmode': 'require'
    }

# Obtener configuración dinámicamente
DATABASE_CONFIG = get_database_config()

# Usuario admin por defecto (hardcodeado para evitar problemas con Railway)
DEFAULT_USER_ID = '119f7084-be9e-416f-81d6-3ffeadb062d5'

# ============================================
# FUNCIONES DE CONEXIÓN
# ============================================

def test_connection():
    """Probar conexión a PostgreSQL"""
    try:
        print(f"🔍 DEBUG: Attempting connection to {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
        with psycopg2.connect(**DATABASE_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"✅ Conexión PostgreSQL exitosa: {version[0]}")
                logger.info(f"🔗 Conectado a: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
                print(f"✅ SUCCESS: Connected to Supabase!")
                return True
    except Exception as e:
        logger.error(f"❌ Error de conexión PostgreSQL: {e}")
        logger.error(f"🔗 Intentando conectar a: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
        print(f"❌ ERROR: {e}")
        return False

@contextmanager
def get_db_connection():
    """Context manager para conexiones de base de datos"""
    conn = None
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        conn.autocommit = False
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error de base de datos: {e}")
        raise
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor(dict_cursor=True):
    """Context manager para cursores de base de datos"""
    with get_db_connection() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en operación de base de datos: {e}")
            raise
        finally:
            cursor.close()

# ============================================
# FUNCIONES DE DOCUMENTOS
# ============================================

def create_document(name: str, content: str, size: int, user_id: str = None, metadata: dict = None):
    """Crear un nuevo documento"""
    import uuid
    from datetime import datetime
    
    doc_id = str(uuid.uuid4())
    user_id = user_id or DEFAULT_USER_ID
    metadata = metadata or {}
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            INSERT INTO documents (id, user_id, name, content, size, status, upload_date, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, name, status, upload_date
        """, (doc_id, user_id, name, content, size, 'processed', datetime.now(), 
              psycopg2.extras.Json(metadata)))
        
        result = cursor.fetchone()
        logger.info(f"✅ Documento creado: {result['name']}")
        return dict(result)

def get_documents(user_id: str = None):
    """Obtener todos los documentos"""
    user_id = user_id or DEFAULT_USER_ID
    
    print(f"🔍 DEBUG: Getting documents for user {user_id}")
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT id, name, size, status, upload_date, metadata
            FROM documents 
            WHERE user_id = %s 
            ORDER BY upload_date DESC
        """, (user_id,))
        
        documents = cursor.fetchall()
        print(f"🔍 DEBUG: Found {len(documents)} documents")
        return [dict(doc) for doc in documents]

def get_document_content(doc_id: str, user_id: str = None):
    """Obtener contenido de un documento específico"""
    user_id = user_id or DEFAULT_USER_ID
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT id, name, content, size, metadata
            FROM documents 
            WHERE id = %s AND user_id = %s
        """, (doc_id, user_id))
        
        document = cursor.fetchone()
        return dict(document) if document else None

def delete_document(doc_id: str, user_id: str = None):
    """Eliminar un documento"""
    user_id = user_id or DEFAULT_USER_ID
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            DELETE FROM documents 
            WHERE id = %s AND user_id = %s
            RETURNING name
        """, (doc_id, user_id))
        
        deleted = cursor.fetchone()
        if deleted:
            logger.info(f"✅ Documento eliminado: {deleted['name']}")
            return True
        return False

# ============================================
# FUNCIONES DE CONVERSACIONES
# ============================================

def create_conversation(question: str, answer: str, sources: str = None, 
                       user_id: str = None, ai_model: str = 'gpt-3.5-turbo'):
    """Crear una nueva conversación"""
    import uuid
    from datetime import datetime
    
    conv_id = str(uuid.uuid4())
    user_id = user_id or DEFAULT_USER_ID
    
    print(f"🔍 DEBUG: Creating conversation with user_id: {user_id}")
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            INSERT INTO conversations (id, user_id, question, answer, sources, timestamp, ai_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, timestamp
        """, (conv_id, user_id, question, answer, sources, datetime.now(), ai_model))
        
        result = cursor.fetchone()
        logger.info(f"✅ Conversación guardada: ID {result['id']}")
        print(f"✅ SUCCESS: Conversation saved with ID {result['id']}")
        return dict(result)

def get_conversations(user_id: str = None, limit: int = 50):
    """Obtener conversaciones recientes"""
    user_id = user_id or DEFAULT_USER_ID
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT id, question, answer, sources, timestamp, ai_model
            FROM conversations 
            WHERE user_id = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (user_id, limit))
        
        conversations = cursor.fetchall()
        return [dict(conv) for conv in conversations]

def get_conversation_history(user_id: str = None):
    """Obtener historial completo de conversaciones"""
    user_id = user_id or DEFAULT_USER_ID
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT question, answer, timestamp
            FROM conversations 
            WHERE user_id = %s 
            ORDER BY timestamp ASC
        """, (user_id,))
        
        history = cursor.fetchall()
        return [dict(conv) for conv in history]

# ============================================
# FUNCIONES DE USUARIOS
# ============================================

def get_user_stats(user_id: str = None):
    """Obtener estadísticas del usuario"""
    user_id = user_id or DEFAULT_USER_ID
    
    with get_db_cursor() as cursor:
        # Contar documentos
        cursor.execute("SELECT COUNT(*) as doc_count FROM documents WHERE user_id = %s", (user_id,))
        doc_count = cursor.fetchone()['doc_count']
        
        # Contar conversaciones
        cursor.execute("SELECT COUNT(*) as conv_count FROM conversations WHERE user_id = %s", (user_id,))
        conv_count = cursor.fetchone()['conv_count']
        
        # Tamaño total de documentos
        cursor.execute("SELECT COALESCE(SUM(size), 0) as total_size FROM documents WHERE user_id = %s", (user_id,))
        total_size = cursor.fetchone()['total_size']
        
        return {
            'documents': doc_count,
            'conversations': conv_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

# ============================================
# INICIALIZACIÓN
# ============================================

def initialize_database():
    """Inicializar y verificar la base de datos"""
    logger.info("🔌 Inicializando conexión a PostgreSQL...")
    
    # Mostrar configuración actual
    print(f"🔍 DEBUG: Database config: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
    print(f"🔍 DEBUG: Default user ID: {DEFAULT_USER_ID}")
    logger.info(f"🔗 Usando base de datos: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
    
    if not DATABASE_CONFIG['password']:
        logger.error("❌ ERROR: Password no configurado en variables de entorno")
        return False
    
    if test_connection():
        logger.info("✅ Base de datos PostgreSQL lista")
        return True
    else:
        logger.error("❌ No se pudo conectar a PostgreSQL")
        return False

# Probar conexión al importar el módulo
if __name__ == "__main__":
    initialize_database()
    
    # Mostrar estadísticas
    stats = get_user_stats()
    print(f"📊 Estadísticas del usuario admin:")
    print(f"   📄 Documentos: {stats['documents']}")
    print(f"   💬 Conversaciones: {stats['conversations']}")
    print(f"   💾 Tamaño total: {stats['total_size_mb']} MB")