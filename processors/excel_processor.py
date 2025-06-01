"""
📊 Procesador especializado para archivos Excel
Maneja extracción de texto, análisis de datos y generación de gráficas
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from typing import Dict, Any, Optional
import logging
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class ExcelProcessor(BaseProcessor):
    """
    Procesador especializado para archivos Excel (.xlsx, .xls)
    """
    
    def __init__(self):
        super().__init__(
            supported_extensions=[".xlsx", ".xls"],
            processor_name="Excel Processor"
        )
        self.max_rows = 200
        self.max_cols = 50
    
    async def extract_text(self, file_content: bytes, filename: str) -> str:
        """
        Extrae texto estructurado de archivos Excel
        """
        try:
            # Leer Excel con pandas
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)
            
            # Aplicar límites para optimización
            if len(df) > self.max_rows:
                df = df.head(self.max_rows)
                logger.info(f"📊 Limited to {self.max_rows} rows for performance")
            
            if len(df.columns) > self.max_cols:
                df = df.iloc[:, :self.max_cols]
                logger.info(f"📊 Limited to {self.max_cols} columns for performance")
            
            # Convertir a texto estructurado
            text_content = self._dataframe_to_text(df, filename)
            
            logger.info(f"✅ Extracted text from Excel: {len(text_content)} characters")
            return text_content
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from Excel: {e}")
            raise Exception(f"Error reading Excel file: {str(e)}")
    
    def _dataframe_to_text(self, df: pd.DataFrame, filename: str) -> str:
        """Convierte DataFrame a texto estructurado para RAG"""
        lines = []
        
        # Header con información del archivo
        lines.append(f"ARCHIVO EXCEL: {filename}")
        lines.append(f"DIMENSIONES: {len(df)} filas x {len(df.columns)} columnas")
        lines.append(f"COLUMNAS: {', '.join(df.columns.astype(str))}")
        lines.append("")
        
        # Encabezados de columnas
        lines.append("DATOS DEL EXCEL:")
        lines.append(" | ".join(df.columns.astype(str)))
        lines.append("-" * 50)
        
        # Contenido de las filas
        for idx, row in df.iterrows():
            row_text = " | ".join([str(val) if pd.notna(val) else "vacío" for val in row])
            lines.append(f"Fila {idx + 1}: {row_text}")
        
        # Resumen estadístico para columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            lines.append("")
            lines.append("RESUMEN ESTADÍSTICO:")
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    lines.append(f"{col}: min={col_data.min():.2f}, max={col_data.max():.2f}, promedio={col_data.mean():.2f}")
        
        return "\n".join(lines)
    
    def analyze_content(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Análisis avanzado del contenido Excel
        """
        try:
            # Re-leer el archivo para análisis (esto se optimizaría en producción)
            # Por ahora, generamos análisis básico desde el contenido extraído
            
            analysis = {
                "type": "excel_analysis",
                "filename": filename,
                "insights": self._generate_insights_from_text(content),
                "charts_available": True,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"📊 Generated analysis for {filename}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Excel content: {e}")
            return None
    
    def _generate_insights_from_text(self, content: str) -> list:
        """Genera insights básicos desde el texto extraído"""
        insights = []
        lines = content.split('\n')
        
        # Extraer información básica
        for line in lines:
            if line.startswith("DIMENSIONES:"):
                insights.append(f"📊 {line}")
            elif line.startswith("COLUMNAS:"):
                cols = line.replace("COLUMNAS: ", "")
                col_count = len(cols.split(", "))
                insights.append(f"📋 Dataset contiene {col_count} columnas de datos")
        
        # Buscar información estadística
        stat_section = False
        for line in lines:
            if line.startswith("RESUMEN ESTADÍSTICO:"):
                stat_section = True
                continue
            elif stat_section and ":" in line and "min=" in line:
                insights.append(f"📈 {line}")
        
        return insights[:10]  # Limitar a 10 insights
    
    async def generate_charts(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Genera gráficas automáticas del archivo Excel
        """
        try:
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)
            
            # Aplicar límites
            if len(df) > self.max_rows:
                df = df.head(self.max_rows)
            if len(df.columns) > self.max_cols:
                df = df.iloc[:, :self.max_cols]
            
            charts = []
            numeric_cols = df.select_dtypes(include=['number']).columns
            text_cols = df.select_dtypes(include=['object']).columns
            
            # 1. Histograma de primera columna numérica
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, title=f"Distribución de {col}")
                charts.append({
                    "type": "histogram",
                    "title": f"Distribución de {col}",
                    "chart_json": fig.to_json()
                })
            
            # 2. Gráfica de barras de primera columna categórica
            if len(text_cols) > 0:
                col = text_cols[0]
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 10 valores en {col}")
                charts.append({
                    "type": "bar",
                    "title": f"Top 10 valores en {col}",
                    "chart_json": fig.to_json()
                })
            
            # 3. Scatter plot si hay 2+ columnas numéricas
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(df, x=x_col, y=y_col,
                               title=f"Relación {x_col} vs {y_col}")
                charts.append({
                    "type": "scatter",
                    "title": f"Relación {x_col} vs {y_col}",
                    "chart_json": fig.to_json()
                })
            
            logger.info(f"📊 Generated {len(charts)} charts for {filename}")
            return {
                "success": True,
                "charts": charts,
                "total_charts": len(charts)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating charts: {e}")
            return {
                "success": False,
                "error": str(e),
                "charts": []
            }
    
    def validate_file(self, file_content: bytes, filename: str) -> tuple[bool, str]:
        """Validación específica para archivos Excel"""
        # Validación base
        is_valid, message = super().validate_file(file_content, filename)
        if not is_valid:
            return is_valid, message
        
        # Validación específica de Excel
        try:
            # Intentar leer el archivo
            pd.read_excel(io.BytesIO(file_content), sheet_name=0, nrows=1)
            return True, "Valid Excel file"
        except Exception as e:
            return False, f"Invalid Excel file: {str(e)}"