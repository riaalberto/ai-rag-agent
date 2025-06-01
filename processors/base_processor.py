"""
🏗️ Clase base para todos los procesadores de archivos
Establece la interfaz común que deben implementar todos los procesadores
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """
    Clase base abstracta para procesadores de archivos
    """
    
    def __init__(self, supported_extensions: List[str], processor_name: str):
        self.supported_extensions = supported_extensions
        self.processor_name = processor_name
        logger.info(f"✅ {processor_name} processor initialized")
    
    def can_process(self, filename: str) -> bool:
        """Verifica si este procesador puede manejar el archivo"""
        extension = filename.lower().split('.')[-1]
        return f".{extension}" in self.supported_extensions
    
    @abstractmethod
    async def extract_text(self, file_content: bytes, filename: str) -> str:
        """
        Extrae texto del archivo (método obligatorio para todos los procesadores)
        
        Args:
            file_content: Contenido del archivo en bytes
            filename: Nombre del archivo
            
        Returns:
            str: Texto extraído del archivo
        """
        pass
    
    def analyze_content(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Análisis opcional del contenido (pueden sobreescribir los procesadores que lo necesiten)
        
        Args:
            content: Texto extraído
            filename: Nombre del archivo
            
        Returns:
            Dict opcional con análisis adicional
        """
        return None
    
    def generate_metadata(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Genera metadatos básicos del archivo
        
        Args:
            file_content: Contenido del archivo
            filename: Nombre del archivo
            
        Returns:
            Dict con metadatos básicos
        """
        return {
            "filename": filename,
            "file_size": len(file_content),
            "processor_used": self.processor_name,
            "supported_extensions": self.supported_extensions
        }
    
    async def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Método principal que procesa el archivo completo
        
        Args:
            file_content: Contenido del archivo
            filename: Nombre del archivo
            
        Returns:
            Dict con toda la información procesada
        """
        try:
            logger.info(f"🔄 Processing {filename} with {self.processor_name}")
            
            # 1. Extraer texto
            extracted_text = await self.extract_text(file_content, filename)
            
            # 2. Generar metadatos
            metadata = self.generate_metadata(file_content, filename)
            
            # 3. Análisis opcional
            analysis = self.analyze_content(extracted_text, filename)
            
            result = {
                "success": True,
                "extracted_text": extracted_text,
                "metadata": metadata,
                "analysis": analysis,
                "processor": self.processor_name
            }
            
            logger.info(f"✅ Successfully processed {filename}")
            return result
            
        except Exception as e:
            error_msg = f"❌ Error processing {filename} with {self.processor_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "processor": self.processor_name
            }
    
    def validate_file(self, file_content: bytes, filename: str) -> tuple[bool, str]:
        """
        Valida si el archivo es válido para este procesador
        
        Args:
            file_content: Contenido del archivo
            filename: Nombre del archivo
            
        Returns:
            Tuple (es_válido, mensaje)
        """
        # Validación básica de extensión
        if not self.can_process(filename):
            return False, f"Extension not supported by {self.processor_name}"
        
        # Validación básica de tamaño
        if len(file_content) == 0:
            return False, "File is empty"
        
        # Validación básica de tamaño máximo (50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            return False, f"File too large (max {max_size/1024/1024}MB)"
        
        return True, "File is valid"

class ProcessorRegistry:
    """
    Registro centralizado de todos los procesadores disponibles
    """
    
    def __init__(self):
        self.processors: List[BaseProcessor] = []
        logger.info("🏗️ Processor registry initialized")
    
    def register(self, processor: BaseProcessor):
        """Registra un nuevo procesador"""
        self.processors.append(processor)
        logger.info(f"📝 Registered processor: {processor.processor_name}")
    
    def get_processor(self, filename: str) -> Optional[BaseProcessor]:
        """Encuentra el procesador adecuado para un archivo"""
        for processor in self.processors:
            if processor.can_process(filename):
                logger.info(f"🎯 Found processor {processor.processor_name} for {filename}")
                return processor
        
        logger.warning(f"❌ No processor found for {filename}")
        return None
    
    def list_supported_extensions(self) -> List[str]:
        """Lista todas las extensiones soportadas"""
        extensions = []
        for processor in self.processors:
            extensions.extend(processor.supported_extensions)
        return sorted(list(set(extensions)))
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Información de todos los procesadores registrados"""
        return {
            "total_processors": len(self.processors),
            "processors": [
                {
                    "name": p.processor_name,
                    "extensions": p.supported_extensions
                }
                for p in self.processors
            ],
            "total_extensions": len(self.list_supported_extensions()),
            "supported_extensions": self.list_supported_extensions()
        }