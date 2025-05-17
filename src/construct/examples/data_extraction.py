"""
Data extraction pipeline using AWS Bedrock Construct API.

This module provides a complete example of a data extraction pipeline
that processes documents to extract structured information.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..advanced.complex_schema import ComplexConstructClient


class DataExtractionPipeline:
    """
    A pipeline for extracting structured data from documents.
    
    This pipeline processes text documents and extracts structured information
    based on a provided JSON schema, with support for batch processing and
    error handling.
    """
    
    def __init__(
        self,
        model_id: str,
        schema_path: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        max_workers: int = 5,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data extraction pipeline.
        
        Args:
            model_id: The Bedrock model identifier
            schema_path: Path to JSON schema file (alternative to schema)
            schema: JSON schema dictionary (alternative to schema_path)
            max_workers: Maximum number of concurrent extraction workers
            profile_name: AWS profile name
            region_name: AWS region name
            logger: Optional logger instance
        """
        if not schema and not schema_path:
            raise ValueError("Either schema or schema_path must be provided")
        
        self.model_id = model_id
        self.schema_path = schema_path
        self.schema = schema
        self.max_workers = max_workers
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Load schema if path provided
        if schema_path and not schema:
            if not os.path.exists(schema_path):
                raise ValueError(f"Schema file not found: {schema_path}")
            
            try:
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema file: {str(e)}")
        
        # Create client
        self.client = ComplexConstructClient(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            logger=self.logger
        )
        
        # Track metrics
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def process_document(
        self,
        document: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single document to extract structured data.
        
        Args:
            document: The document text to process
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            document_id: Optional identifier for the document
            
        Returns:
            Dictionary with extraction results and metadata
        """
        start_time = time.time()
        doc_id = document_id or f"doc-{int(start_time)}"
        
        self.logger.debug(f"Processing document {doc_id}")
        
        try:
            # Extract structured data
            extracted_data = self.client.generate_structured(
                input_text=document,
                schema=self.schema,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Document successfully processed
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.processed_count += 1
            self.success_count += 1
            
            result = {
                "document_id": doc_id,
                "status": "success",
                "data": extracted_data,
                "processing_time": processing_time
            }
            
            self.logger.debug(f"Successfully processed document {doc_id} in {processing_time:.2f}s")
            
        except Exception as e:
            # Error extracting data
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.processed_count += 1
            self.failure_count += 1
            
            result = {
                "document_id": doc_id,
                "status": "error",
                "error": str(e),
                "processing_time": processing_time
            }
            
            self.logger.warning(f"Error processing document {doc_id}: {str(e)}")
        
        return result
    
    def process_documents(
        self,
        documents: List[str],
        max_tokens: int = 4000,
        temperature: float = 0.2,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel.
        
        Args:
            documents: List of document texts to process
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            document_ids: Optional list of document identifiers
            
        Returns:
            List of dictionaries with extraction results
        """
        start_time = time.time()
        
        if document_ids and len(document_ids) != len(documents):
            raise ValueError("If provided, document_ids must have the same length as documents")
        
        # Create document IDs if not provided
        doc_ids = document_ids or [f"doc-{i}" for i in range(len(documents))]
        
        self.logger.info(f"Processing {len(documents)} documents with up to {self.max_workers} workers")
        
        results = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_doc_id = {
                executor.submit(
                    self.process_document, 
                    document, 
                    max_tokens, 
                    temperature, 
                    doc_id
                ): doc_id 
                for document, doc_id in zip(documents, doc_ids)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_doc_id):
                doc_id = future_to_doc_id[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle unexpected errors
                    results.append({
                        "document_id": doc_id,
                        "status": "error",
                        "error": f"Unexpected error: {str(e)}",
                        "processing_time": 0
                    })
                    self.logger.error(f"Unexpected error processing document {doc_id}: {str(e)}")
        
        # Sort results by document ID to maintain original order
        results.sort(key=lambda x: doc_ids.index(x["document_id"]))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log summary
        self.logger.info(
            f"Processed {len(documents)} documents in {total_time:.2f}s: "
            f"{self.success_count} successful, {self.failure_count} failed"
        )
        
        return results
    
    def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Process a text file to extract structured data.
        
        Args:
            input_file: Path to input text file
            output_file: Optional path to save extraction results
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with extraction results
        """
        # Check if input file exists
        if not os.path.exists(input_file):
            raise ValueError(f"Input file not found: {input_file}")
        
        self.logger.info(f"Processing file: {input_file}")
        
        # Read input file
        try:
            with open(input_file, 'r') as f:
                document = f.read()
        except Exception as e:
            raise ValueError(f"Error reading input file: {str(e)}")
        
        # Process document
        result = self.process_document(
            document=document,
            max_tokens=max_tokens,
            temperature=temperature,
            document_id=os.path.basename(input_file)
        )
        
        # Save output if requested
        if output_file and result["status"] == "success":
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            try:
                with open(output_file, 'w') as f:
                    json.dump(result["data"], f, indent=2)
                    
                self.logger.info(f"Results saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving results: {str(e)}")
        
        return result
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        file_pattern: str = "*.txt",
        max_tokens: int = 4000,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Process all text files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Optional directory to save extraction results
            file_pattern: Glob pattern to match input files
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with processing summary and results
        """
        import glob
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Find matching files
        pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {pattern}")
            return {"status": "warning", "message": "No matching files found", "results": []}
        
        self.logger.info(f"Found {len(files)} files to process")
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read all files
        documents = []
        doc_ids = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                documents.append(content)
                doc_ids.append(os.path.basename(file_path))
                
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {str(e)}")
        
        # Process documents
        results = self.process_documents(
            documents=documents,
            max_tokens=max_tokens,
            temperature=temperature,
            document_ids=doc_ids
        )
        
        # Save results if output directory provided
        if output_dir:
            for result in results:
                if result["status"] == "success":
                    doc_id = result["document_id"]
                    output_file = os.path.join(output_dir, f"{os.path.splitext(doc_id)[0]}.json")
                    
                    try:
                        with open(output_file, 'w') as f:
                            json.dump(result["data"], f, indent=2)
                    except Exception as e:
                        self.logger.error(f"Error saving results for {doc_id}: {str(e)}")
        
        # Create summary
        summary = {
            "total_files": len(files),
            "processed_files": len(results),
            "successful_extractions": sum(1 for r in results if r["status"] == "success"),
            "failed_extractions": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
        
        return summary
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics for the pipeline.
        
        Returns:
            Dictionary with metrics
        """
        client_metrics = self.client.get_metrics()
        
        return {
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "request_count": client_metrics["request_count"],
            "token_count": client_metrics["token_count"],
            "error_count": client_metrics["error_count"]
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create an extraction pipeline
    pipeline = DataExtractionPipeline(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        schema_path="../schemas/person.json"
    )
    
    # Example documents
    documents = [
        "John Smith is a 35-year-old software engineer from Seattle, Washington. He has a Bachelor's degree in Computer Science from the University of Washington and speaks English (native) and Spanish (intermediate). His email is john.smith@example.com and his phone number is 555-123-4567.",
        
        "Dr. Jane Doe is the Chief Medical Officer at Boston General Hospital. She is 42 years old and graduated from Harvard Medical School in 2008 after completing her undergraduate studies in Biology at MIT. Dr. Doe is fluent in English, French, and Mandarin Chinese. You can reach her at jane.doe@bostongeneral.org or at her office number 617-555-7890.",
        
        "Michael Johnson (29) works as a marketing specialist for a major tech company. He studied Business Administration at UCLA and has skills in digital marketing, social media management, and data analysis. Michael is based in Austin, Texas and can be contacted at michael.j@techemail.com."
    ]
    
    # Process documents
    results = pipeline.process_documents(documents)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nDocument {i+1} Result:")
        if result["status"] == "success":
            print(f"Status: {result['status']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print("Extracted Data:")
            print(json.dumps(result["data"], indent=2))
        else:
            print(f"Status: {result['status']}")
            print(f"Error: {result['error']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
    
    # Get metrics
    metrics = pipeline.get_metrics()
    print("\nPipeline Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")