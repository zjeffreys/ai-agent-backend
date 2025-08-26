import os
import io
import re
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import HTTPException, UploadFile
import PyPDF2
import docx

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class SensitiveDataPattern:
    """Represents a pattern for detecting sensitive data"""
    pattern_type: str  # "ssn", "phone", "email", "credit_card", "custom"
    regex_pattern: str
    description: str
    risk_level: str  # "high", "medium", "low"

class ProcessedDocument:
    """Represents a processed document with its metadata"""
    def __init__(self, filename: str, content: str, sensitive_data_found: List[SensitiveDataPattern], 
                 processing_summary: Dict[str, Any], anonymized_content: str):
        self.filename = filename
        self.content = content
        self.sensitive_data_found = sensitive_data_found
        self.processing_summary = processing_summary
        self.anonymized_content = anonymized_content

class DocumentProcessor:
    """Main class for processing documents and detecting sensitive data"""
    
    def __init__(self):
        # Predefined sensitive data patterns
        self.patterns = [
            SensitiveDataPattern(
                pattern_type="ssn",
                regex_pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                description="Social Security Number",
                risk_level="high"
            ),
            SensitiveDataPattern(
                pattern_type="phone",
                regex_pattern=r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
                description="Phone Number",
                risk_level="medium"
            ),
            SensitiveDataPattern(
                pattern_type="email",
                regex_pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                description="Email Address",
                risk_level="medium"
            ),
            SensitiveDataPattern(
                pattern_type="credit_card",
                regex_pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                description="Credit Card Number",
                risk_level="high"
            ),
            SensitiveDataPattern(
                pattern_type="address",
                regex_pattern=r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b",
                description="Street Address",
                risk_level="medium"
            ),
            SensitiveDataPattern(
                pattern_type="name",
                regex_pattern=r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                description="Full Name",
                risk_level="medium"
            ),
            SensitiveDataPattern(
                pattern_type="date_of_birth",
                regex_pattern=r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b",
                description="Date of Birth",
                risk_level="high"
            )
        ]
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from Word document"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from plain text file"""
        try:
            return file_content.decode('utf-8').strip()
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1').strip()
            except Exception as e:
                logger.error(f"Error decoding text file: {e}")
                return ""
    
    def detect_sensitive_data(self, text: str) -> List[SensitiveDataPattern]:
        """Detect sensitive data patterns in text"""
        found_patterns = []
        
        for pattern in self.patterns:
            matches = re.finditer(pattern.regex_pattern, text, re.IGNORECASE)
            if matches:
                # Create a copy with actual matches found
                pattern_copy = SensitiveDataPattern(
                    pattern_type=pattern.pattern_type,
                    regex_pattern=pattern.regex_pattern,
                    description=pattern.description,
                    risk_level=pattern.risk_level
                )
                found_patterns.append(pattern_copy)
        
        return found_patterns
    
    def anonymize_text(self, text: str, anonymization_type: str = "redact") -> str:
        """Anonymize sensitive data in text"""
        anonymized_text = text
        
        for pattern in self.patterns:
            if anonymization_type == "redact":
                anonymized_text = re.sub(
                    pattern.regex_pattern, 
                    f"[{pattern.pattern_type.upper()}_REDACTED]", 
                    anonymized_text, 
                    flags=re.IGNORECASE
                )
            elif anonymization_type == "mask":
                anonymized_text = re.sub(
                    pattern.regex_pattern, 
                    "*" * 10, 
                    anonymized_text, 
                    flags=re.IGNORECASE
                )
            elif anonymization_type == "anonymize":
                anonymized_text = re.sub(
                    pattern.regex_pattern, 
                    self._generate_fake_data(pattern.pattern_type), 
                    anonymized_text, 
                    flags=re.IGNORECASE
                )
        
        return anonymized_text
    
    def _generate_fake_data(self, pattern_type: str) -> str:
        """Generate fake data for anonymization"""
        fake_data_map = {
            "ssn": "XXX-XX-XXXX",
            "phone": "(555) 123-4567",
            "email": "user@example.com",
            "credit_card": "****-****-****-1234",
            "address": "123 Main Street",
            "name": "John Doe",
            "date_of_birth": "01/01/1990"
        }
        return fake_data_map.get(pattern_type, "[ANONYMIZED]")
    
    def process_single_document(self, file: UploadFile, anonymization_type: str = "redact") -> ProcessedDocument:
        """Process a single document and return processed data"""
        if not file.filename:
            raise ValueError("File must have a filename")
        
        # Read file content
        content = file.file.read()
        
        # Extract text based on file type
        text = ""
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.pdf'):
            text = self.extract_text_from_pdf(content)
        elif filename_lower.endswith('.docx'):
            text = self.extract_text_from_docx(content)
        elif filename_lower.endswith('.txt'):
            text = self.extract_text_from_txt(content)
        else:
            raise ValueError(f"Unsupported file type: {file.filename}")
        
        if not text:
            raise ValueError(f"Could not extract text from {file.filename}")
        
        # Detect sensitive data
        sensitive_data = self.detect_sensitive_data(text)
        
        # Anonymize content
        anonymized_content = self.anonymize_text(text, anonymization_type)
        
        # Create processing summary
        processing_summary = {
            "file_size": len(content),
            "text_length": len(text),
            "sensitive_patterns_found": len(sensitive_data),
            "anonymization_applied": anonymization_type,
            "processing_timestamp": datetime.now().isoformat(),
            "file_type": filename_lower.split('.')[-1]
        }
        
        return ProcessedDocument(
            filename=file.filename,
            content=text,
            sensitive_data_found=sensitive_data,
            processing_summary=processing_summary,
            anonymized_content=anonymized_content
        )
    
    def process_multiple_documents(self, files: List[UploadFile], 
                                 anonymization_type: str = "redact") -> List[ProcessedDocument]:
        """Process multiple documents"""
        processed_docs = []
        
        for file in files:
            try:
                processed_doc = self.process_single_document(file, anonymization_type)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                # Continue with other files instead of failing completely
                continue
        
        return processed_docs

# Initialize global document processor instance
document_processor = DocumentProcessor()
