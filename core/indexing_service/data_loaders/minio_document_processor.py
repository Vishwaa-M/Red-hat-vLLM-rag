import logging
import os
import asyncio
import json
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from dotenv import load_dotenv
import tempfile
from pathlib import Path
import base64
import shutil
import re
import zipfile
import io
from docling.document_converter import (
    DocumentConverter, PdfFormatOption, WordFormatOption
)
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice, EasyOcrOptions
from docling_core.types.doc.document import DoclingDocument, TextItem, TableItem, PictureItem
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table, NarrativeText
import boto3
from botocore.client import Config
import fitz  # PyMuPDF
from PIL import Image
import torch
import torchvision.transforms as T

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = [
    "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("docling").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    A document processor for loading raw files from MinIO S3 and parsing them using Docling as the
    primary parser and Unstructured.io as the fallback. Extracts text, tables, and embedded images
    into three separate arrays with consistent metadata, optionally storing images as base64 or on disk.
    """

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str,
        prefix: str = "",
        secure: bool = False,
        output_dir: str = "scratch",
        store_images_on_disk: bool = True
    ):
        if not all([minio_endpoint, minio_access_key, minio_secret_key, bucket_name]):
            logger.error("Missing MinIO parameters")
            raise ValueError("MinIO endpoint, access key, secret key, and bucket name required")

        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.secure = secure
        self.supported_formats = ['.pdf', '.docx']
        self.store_images_on_disk = store_images_on_disk
        self.output_dir = Path(output_dir).resolve()
        self.logs_dir = self.output_dir / "logs"
        self.image_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.store_images_on_disk:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        self.temp_files = []
        self.figures_dir = Path("figures").resolve()
        self.min_image_size = (100, 100)  # Minimum width, height in pixels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = T.Compose([T.Resize(self.min_image_size)])
        logger.info(f"DocumentProcessor initialized with device: {self.device}")

        # Check dependencies
        self._check_dependencies()

        # Initialize boto3 S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=minio_endpoint,
                aws_access_key_id=minio_access_key,
                aws_secret_access_key=minio_secret_key,
                config=Config(signature_version='s3v4'),
                use_ssl=secure
            )
            logger.info("Initialized boto3 S3 client")
        except Exception as e:
            logger.error("Failed to initialize S3 client: %s", str(e))
            raise RuntimeError("Failed to initialize S3 client")

        # Configure Docling
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            pipeline_options.ocr_options = EasyOcrOptions(lang=['en'])
            pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
            self.doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options),
                }
            )
            logger.info(f"Docling configured with device: {pipeline_options.accelerator_options.device}")
        except Exception as e:
            logger.error("Failed to initialize Docling converter: %s", str(e))
            self.doc_converter = None

        logger.info("Initialized DocumentProcessor: bucket=%s, prefix=%s, output_dir=%s, store_images_on_disk=%s", 
                    bucket_name, prefix, self.output_dir, store_images_on_disk)

    def _check_dependencies(self):
        """Check for required dependencies and log warnings if missing."""
        try:
            import pytesseract
            logger.info("Tesseract found: %s", pytesseract.get_tesseract_version())
        except (ImportError, pytesseract.TesseractNotFoundError):
            logger.warning("Tesseract not found. Required for Unstructured.io OCR.")
        try:
            import fitz
            logger.info("PyMuPDF found")
        except ImportError:
            logger.warning("PyMuPDF not installed. Required for PDF image extraction.")
        try:
            import PIL
            logger.info("Pillow found")
        except ImportError:
            logger.warning("Pillow not installed. Required for image processing.")
        try:
            import torch
            logger.info(f"PyTorch found, device available: {self.device}")
        except ImportError:
            logger.warning("PyTorch not installed. Required for GPU-accelerated image processing.")

    def _create_temp_file(self, content: bytes, source: str) -> str:
        file_extension = os.path.splitext(source)[1]
        if not content:
            raise ValueError(f"Empty content for {source}")
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        self.temp_files.append(temp_file_path)
        logger.debug("Created temp file: %s", temp_file_path)
        return temp_file_path

    def _sanitize_metadata(self, metadata: dict) -> dict:
        sanitized = {
            "element_type": "unknown",
            "source": "",
            "page_number": 1,
            "content_layer": "BODY",
            "level": 0,
            "caption": None
        }
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def _save_elements_to_log(self, text_elements, table_elements, image_elements, source: str):
        base_filename = os.path.basename(source).replace('/', '_')
        json_files = [
            (text_elements, f"text_elements_{base_filename}.json"),
            (table_elements, f"table_elements_{base_filename}.json"),
            (image_elements, f"image_elements_{base_filename}.json")
        ]
        for elements, filename in json_files:
            if elements:
                json_path = self.logs_dir / filename
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(elements, f, indent=2, ensure_ascii=False)
                    logger.info("Saved %d %s elements to %s", len(elements), filename.split('_')[0], json_path)
                    if filename.startswith("text_elements"):
                        element_types = [e[2]["element_type"] for e in elements]
                        logger.info("Text element types for %s: %s", source, set(element_types))
                except Exception as e:
                    logger.error("Failed to save %s: %s", json_path, str(e))
            else:
                logger.debug("No %s elements for %s", filename.split('_')[0], source)

    def _process_image(self, image: Image.Image) -> Image.Image:
        try:
            image_tensor = T.ToTensor()(image).to(self.device)
            processed_tensor = self.transform(image_tensor)
            return T.ToPILImage()(processed_tensor.cpu())
        except Exception as e:
            logger.error("Image processing failed: %s", str(e))
            raise

    def _extract_docx_images(self, temp_file_path: str, source: str, image_elements: List[Tuple[str, str, dict]]) -> int:
        images_saved = 0
        try:
            with zipfile.ZipFile(temp_file_path, 'r') as docx_zip:
                media_files = [f for f in docx_zip.infolist() if f.filename.startswith('word/media/')]
                if not media_files:
                    logger.info("No images found in word/media/ for %s", source)
                for file_info in media_files:
                    image_data = docx_zip.read(file_info.filename)
                    image_ext = os.path.splitext(file_info.filename)[1].lstrip('.').lower()
                    if image_ext in ['png', 'jpg', 'png', 'bmp', 'gif', 'tiff', 'emf', 'wmf']:
                        try:
                            image = Image.open(io.BytesIO(image_data))
                            image = self._process_image(image)  # Use GPU-accelerated processing
                            width, height = image.size
                            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                                logger.debug("Skipping small DOCX image %s: %dx%d", file_info.filename, width, height)
                                continue
                            metadata = self._sanitize_metadata({
                                "element_type": "Image",
                                "source": source,
                                "page_number": 1,
                                "content_layer": "BODY",
                                "level": 0,
                                "caption": None,
                                "width": width,
                                "height": height
                            })
                            if self.store_images_on_disk:
                                image_filename = f"{os.path.basename(temp_file_path).replace('.docx', '')}_img{len(image_elements)}.png"
                                dest_path = self.image_dir / image_filename
                                image.save(dest_path, 'PNG')
                                metadata["image_path"] = str(dest_path)
                                logger.info("Saved DOCX image to %s (%dx%d)", dest_path, width, height)
                            else:
                                buffered = io.BytesIO()
                                image.save(buffered, format="PNG")
                                metadata["image_base64"] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                logger.info("Encoded DOCX image as base64 (%dx%d)", width, height)
                            image_elements.append(("<!-- Image -->", source, metadata))
                            images_saved += 1
                        except Exception as e:
                            logger.warning("Failed to process DOCX image %s: %s", file_info.filename, str(e))
        except Exception as e:
            logger.error("Failed to extract images from DOCX %s: %s", source, str(e))
        return images_saved

    async def load_and_parse_documents(self) -> Tuple[List[Document], List[Document], List[Document]]:
        """
        Load and parse documents from MinIO, returning three separate arrays for text, table, and image documents.

        Returns:
            Tuple[List[Document], List[Document], List[Document]]: Three lists containing text, table, and image Document objects.
        """
        raw_files = await self._load_raw_files()
        if not raw_files:
            logger.info("No documents to parse")
            return [], [], []
        
        from concurrent.futures import ThreadPoolExecutor
        text_documents = []
        table_documents = []
        image_documents = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [self._parse_raw_file(content, source) for content, source in raw_files]
            results = await asyncio.gather(
                *[asyncio.get_event_loop().run_in_executor(executor, lambda: asyncio.run(task)) for task in tasks],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    text_elements, table_elements, image_elements = result
                    # Convert text elements to Document objects
                    for content, source, metadata in text_elements:
                        metadata["content_type"] = "text"
                        text_documents.append(Document(page_content=content, metadata=metadata))
                    # Convert table elements to Document objects
                    for content, source, metadata in table_elements:
                        metadata["content_type"] = "table"
                        table_documents.append(Document(page_content=content, metadata=metadata))
                    # Convert image elements to Document objects
                    for content, source, metadata in image_elements:
                        metadata["content_type"] = "image"
                        image_documents.append(Document(page_content=content, metadata=metadata))
                else:
                    logger.error("Parsing error: %s", str(result))
        
        logger.info("Parsed %d text documents, %d table documents, %d image documents", 
                    len(text_documents), len(table_documents), len(image_documents))
        return text_documents, table_documents, image_documents

    async def _load_raw_files(self) -> List[Tuple[bytes, str]]:
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
            raw_files = []
            if 'Contents' not in response:
                logger.info("No objects found in bucket %s with prefix %s", self.bucket_name, self.prefix)
                return raw_files
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                file_extension = os.path.splitext(key.lower())[1]
                if file_extension not in self.supported_formats:
                    logger.debug("Skipping unsupported file %s", key)
                    continue
                obj_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                content = obj_response['Body'].read()
                source = f"s3://{self.bucket_name}/{key}"
                raw_files.append((content, source))
            logger.info("Loaded %d raw files", len(raw_files))
            return raw_files
        except Exception as e:
            logger.error("Failed to load files: %s", str(e))
            raise RuntimeError(f"Failed to load files: {str(e)}")

    def _table_to_json(self, table_content: str, metadata: dict) -> dict:
        """Convert a pipe-separated table content to JSON format."""
        try:
            # Split into rows, removing empty lines and non-table lines
            rows = [row.strip() for row in table_content.split('\n') if row.strip() and '|' in row]
            if not rows:
                logger.debug("No valid table rows found")
                return {"table_data": [], "metadata": metadata}
            
            # Extract headers from the first row, ignoring empty cells
            headers = [cell.strip() for cell in rows[0].split('|') if cell.strip()]
            if not headers or len(headers) == 1:  # Fallback if headers are invalid
                headers = [f"Column_{i+1}" for i in range(max(len(row.split('|')) for row in rows))]
                logger.warning("Invalid headers, using default: %s", headers)

            # Process data rows (skip header and separator rows)
            table_data = []
            for row in rows[1:]:
                # Skip separator rows (e.g., |----|)
                if all(cell.strip().startswith('-') for cell in row.split('|') if cell.strip()):
                    continue
                cells = [cell.strip() for cell in row.split('|') if cell]
                if len(cells) > 0:
                    # Create a dict for the row, mapping headers to cells
                    row_dict = {headers[min(i, len(headers)-1)]: cells[i] for i in range(len(cells)) if i < len(headers)}
                    table_data.append(row_dict)
            
            logger.debug("Raw table content: %s", table_content[:200])
            return {"table_data": table_data, "metadata": metadata}
        except Exception as e:
            logger.error("Failed to convert table to JSON: %s", str(e))
            return {"table_data": [], "metadata": metadata, "error": str(e)}

    async def _parse_raw_file(self, content: bytes, source: str) -> Tuple[List[Tuple[str, str, dict]], List[Tuple[str, str, dict]], List[Tuple[str, str, dict]]]:
        file_extension = os.path.splitext(source.lower())[1]
        if file_extension not in self.supported_formats:
            logger.warning("Unsupported extension %s for %s", file_extension, source)
            return [], [], []

        text_elements, table_elements, image_elements = [], [], []
        images_saved = 0
        temp_file_path = None

        try:
            temp_file_path = self._create_temp_file(content, source)

            # Primary parser: Docling
            try:
                # Disable tqdm progress bars to avoid '_lock' error
                os.environ["TQDM_DISABLE"] = "1"
                conv_result = self.doc_converter.convert(temp_file_path)
                logger.debug("Docling conversion status for %s: %s", source, conv_result.status)
                if conv_result.status != ConversionStatus.SUCCESS:
                    logger.warning("Docling failed for %s: %s", source, conv_result.errors)
                    raise RuntimeError(f"Docling conversion failed: {conv_result.errors}")

                doc: DoclingDocument = conv_result.document
                text_count, table_count, picture_count = 0, 0, 0
                picture_pages = set()
                for element, level in doc.iterate_items():
                    if isinstance(element, TextItem):
                        content = element.text
                        if not content.strip():
                            logger.debug("Skipping empty text element on page %s", element.prov[0].page_no if element.prov else 1)
                            continue
                        metadata = self._sanitize_metadata({
                            "element_type": str(element.label),
                            "source": source,
                            "page_number": element.prov[0].page_no if element.prov else 1,
                            "content_layer": element.content_layer.value,
                            "level": level,
                            "char_count": len(content)
                        })
                        text_elements.append((content, source, metadata))
                        text_count += 1
                    elif isinstance(element, TableItem):
                        content = element.export_to_markdown(doc=doc)
                        if not content.strip():
                            logger.debug("Skipping empty table element on page %s", element.prov[0].page_no if element.prov else 1)
                            continue
                        page_num = element.prov[0].page_no if element.prov else 1
                        caption = self._find_caption(doc.iterate_items(), page_num, "Table")
                        if caption and content.startswith(caption):
                            content = content[len(caption):].strip()
                        table_rows = len(content.split('\n')) - 1
                        table_cols = len(content.split('\n')[0].split('|')) - 1 if content else 0
                        metadata = self._sanitize_metadata({
                            "element_type": "Table",
                            "source": source,
                            "page_number": page_num,
                            "content_layer": element.content_layer.value,
                            "level": level,
                            "caption": caption,
                            "rows": table_rows,
                            "columns": table_cols
                        })
                        # Convert table to JSON
                        json_content = json.dumps(self._table_to_json(content, metadata))
                        table_elements.append((json_content, source, metadata))
                        table_count += 1
                        logger.debug("Extracted Docling table on page %s: %s", metadata["page_number"], json_content[:100])
                    elif isinstance(element, PictureItem):
                        picture_pages.add(element.prov[0].page_no if element.prov else 1)
                        picture_count += 1

                logger.info("Docling extracted %d text, %d tables, %d pictures from %s", text_count, table_count, picture_count, source)

                # Extract images for PDFs using PyMuPDF
                if file_extension == '.pdf' and picture_count > 0:
                    try:
                        fitz_doc = fitz.open(temp_file_path)
                        for page_num in picture_pages:
                            page = fitz_doc[page_num - 1]
                            image_list = page.get_images(full=True)
                            for img_index, img in enumerate(image_list):
                                try:
                                    xref = img[0]
                                    base_image = fitz_doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    try:
                                        image = Image.open(io.BytesIO(image_bytes))
                                        image = self._process_image(image)  # Use GPU-accelerated processing
                                        width, height = image.size
                                        if width < self.min_image_size[0] or height < self.min_image_size[1]:
                                            logger.debug("Skipping small image on page %d: %dx%d", page_num, width, height)
                                            continue
                                        metadata = self._sanitize_metadata({
                                            "element_type": "Image",
                                            "source": source,
                                            "page_number": page_num,
                                            "content_layer": "BODY",
                                            "level": 0,
                                            "caption": self._find_caption(doc.iterate_items(), page_num, "Figure"),
                                            "width": width,
                                            "height": height
                                        })
                                        if self.store_images_on_disk:
                                            image_filename = f"{os.path.basename(temp_file_path).replace('.pdf', '')}_page{page_num}_{img_index}.png"
                                            dest_path = self.image_dir / image_filename
                                            image.save(dest_path, 'PNG')
                                            metadata["image_path"] = str(dest_path)
                                            logger.info("Saved PyMuPDF image to %s (%dx%d)", dest_path, width, height)
                                        else:
                                            buffered = io.BytesIO()
                                            image.save(buffered, format="PNG")
                                            metadata["image_base64"] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                            logger.info("Encoded PDF image as base64 (%dx%d)", width, height)
                                        image_elements.append(("<!-- Image -->", source, metadata))
                                        images_saved += 1
                                    except Exception as img_error:
                                        logger.error("Failed to process image %d on page %d for %s: %s", img_index, page_num, source, str(img_error))
                                except Exception as img_extract_error:
                                    logger.error("Failed to extract image %d on page %d for %s: %s", img_index, page_num, source, str(img_extract_error))
                        fitz_doc.close()
                    except Exception as e:
                        logger.error("PyMuPDF image extraction failed for %s: %s", source, str(e))

                # Extract images for DOCX
                if file_extension == '.docx':
                    images_saved += self._extract_docx_images(temp_file_path, source, image_elements)

            except Exception as docling_error:
                logger.exception("Docling processing failed for %s: %s", source, str(docling_error))
                text_elements, table_elements, image_elements = [], [], []
                images_saved = 0

                # Fallback: Unstructured.io
                try:
                    unstructured_elements = await asyncio.to_thread(
                        partition,
                        filename=temp_file_path,
                        strategy="hi_res",
                        include_metadata=True,
                        extract_image_block_types=["Image"],
                        extract_tables=True,
                        extract_images_in_pdf=True,
                        languages=["eng"],  # Updated to use languages instead of ocr_languages
                        ocr_device=self.device  # Pass device for OCR
                    )
                    for element in unstructured_elements:
                        if element.category == "Image":
                            content = "<!-- Image -->"
                            image_path = None
                            image_base64 = None
                            try:
                                if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                                    logger.debug("Found image_base64 for page %s", getattr(element.metadata, "page_number", 1))
                                    image_data = base64.b64decode(element.metadata.image_base64)
                                    image = Image.open(io.BytesIO(image_data))
                                    image = self._process_image(image)  # Use GPU-accelerated processing
                                    width, height = image.size
                                    if width < self.min_image_size[0] or height < self.min_image_size[1]:
                                        logger.debug("Skipping small Unstructured.io image on page %s: %dx%d", 
                                                     getattr(element.metadata, "page_number", 1), width, height)
                                        continue
                                    if self.store_images_on_disk:
                                        image_filename = f"{os.path.basename(temp_file_path).replace(file_extension, '')}_img{len(image_elements)}.png"
                                        dest_path = self.image_dir / image_filename
                                        image.save(dest_path, 'PNG')
                                        image_path = str(dest_path)
                                        logger.info("Saved Unstructured.io image to %s (%dx%d)", image_path, width, height)
                                    else:
                                        buffered = io.BytesIO()
                                        image.save(buffered, format="PNG")
                                        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                        logger.info("Encoded Unstructured.io image as base64 (%dx%d)", width, height)
                                    images_saved += 1
                                else:
                                    logger.warning("No image_base64 found for image element on page %s", getattr(element.metadata, "page_number", 1))
                            except Exception as img_error:
                                logger.error("Failed to decode Unstructured.io image on page %s: %s", getattr(element.metadata, "page_number", 1), str(img_error))
                            metadata = self._sanitize_metadata({
                                "element_type": "Image",
                                "source": source,
                                "page_number": getattr(element.metadata, "page_number", 1),
                                "content_layer": "BODY",
                                "level": 0,
                                "image_path": image_path,
                                "image_base64": image_base64,
                                "caption": self._find_caption(unstructured_elements, getattr(element.metadata, "page_number", 1), "Figure"),
                                "width": width if 'width' in locals() else None,
                                "height": height if 'height' in locals() else None
                            })
                            image_elements.append((content, source, metadata))
                        elif element.category == "Table":
                            content = str(element.text)
                            if not content.strip():
                                logger.debug("Skipping empty table element on page %s", getattr(element.metadata, "page_number", 1))
                                continue
                            page_num = getattr(element.metadata, "page_number", 1)
                            caption = self._find_caption(unstructured_elements, page_num, "Table")
                            if caption and content.startswith(caption):
                                content = content[len(caption):].strip()
                            table_rows = len(content.split('\n')) - 1
                            table_cols = len(content.split('\n')[0].split('|')) - 1 if content else 0
                            metadata = self._sanitize_metadata({
                                "element_type": "Table",
                                "source": source,
                                "page_number": page_num,
                                "content_layer": "BODY",
                                "level": 0,
                                "caption": caption,
                                "rows": table_rows,
                                "columns": table_cols
                            })
                            # Convert table to JSON
                            json_content = json.dumps(self._table_to_json(content, metadata))
                            table_elements.append((json_content, source, metadata))
                            logger.debug("Extracted Unstructured.io table on page %s: %s", metadata["page_number"], json_content[:100])
                        else:
                            content = str(element.text)
                            if not content.strip():
                                logger.debug("Skipping empty text element on page %s", getattr(element.metadata, "page_number", 1))
                                continue
                            metadata = self._sanitize_metadata({
                                "element_type": element.category,
                                "source": source,
                                "page_number": getattr(element.metadata, "page_number", 1),
                                "content_layer": "BODY",
                                "level": 0,
                                "char_count": len(content)
                            })
                            text_elements.append((content, source, metadata))
                            logger.debug("Extracted Unstructured.io text on page %s: %s", metadata["page_number"], content[:100])
                except Exception as unstructured_error:
                    logger.error("Unstructured.io failed for %s: %s", source, str(unstructured_error))
                    text_elements.append(("", source, self._sanitize_metadata({
                        "element_type": "Error",
                        "source": source,
                        "page_number": 1,
                        "content_layer": "BODY",
                        "level": 0,
                        "error": str(unstructured_error),
                        "char_count": 0
                    })))

            self._save_elements_to_log(text_elements, table_elements, image_elements, source)
            logger.info("Parsed %d text, %d tables, %d images from %s (saved %d images)",
                        len(text_elements), len(table_elements), len(image_elements), source, images_saved)
            return text_elements, table_elements, image_elements

        except Exception as e:
            logger.error("Parsing failed for %s: %s", source, str(e))
            metadata = self._sanitize_metadata({
                "element_type": "Error",
                "source": "",
                "page_number": 1,
                "content_layer": "BODY",
                "level": 0,
                "error": str(e),
                "char_count": 0
            })
            return [("", source, metadata)], [], []

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    self.temp_files.remove(temp_file_path)
                    logger.debug("Cleaned up temp file: %s", temp_file_path)
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file %s: %s", temp_file_path, str(cleanup_error))

    def _find_caption(self, elements, page_number: int, element_type: str) -> Optional[str]:
        # Enhanced caption detection: search for captions on same or adjacent pages, considering proximity
        caption_pattern = rf"(?:{element_type}|Fig\.|Figure|Table)\s*\d+\s*[:.-]\s*.*?(?:\n|$)"
        for element in elements:
            if isinstance(element, (TextItem, NarrativeText, tuple)):
                text = element.text if isinstance(element, (TextItem, NarrativeText)) else element[0]
                page_no = None
                if isinstance(element, TextItem) and hasattr(element, 'prov') and element.prov:
                    page_no = element.prov[0].page_no
                elif isinstance(element, NarrativeText) and hasattr(element.metadata, 'page_number'):
                    page_no = element.metadata.page_number
                elif isinstance(element, tuple) and len(element) > 2:
                    page_no = element[2].get('page_number', None)
                if page_no in [page_number - 1, page_number, page_number + 1]:
                    match = re.search(caption_pattern, text, re.IGNORECASE)
                    if match:
                        caption = match.group(0).strip()
                        logger.debug("Found %s caption on page %d: %s", element_type, page_no, caption)
                        return caption
                    # Check for captions in nearby headers or paragraphs
                    if isinstance(element, TextItem) and element.label in ["header", ""]:
                        if "caption" in text.lower() or element_type.lower() in text.lower():
                            logger.debug("Found potential %s caption in header/paragraph on page %d: %s", element_type, page_no, text)
                            return text.strip()
        logger.debug("No %s caption found for page %d", element_type, page_number)
        return None

    def cleanup(self):
        for temp_file in self.temp_files[:]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug("Removed temp file: %s", temp_file)
                except Exception as e:
                    logger.warning("Failed to cleanup temp file %s: %s", temp_file, str(e))
            self.temp_files.remove(temp_file)
        if self.figures_dir.exists():
            try:
                shutil.rmtree(self.figures_dir)
                logger.debug("Cleaned up figures/ directory")
            except Exception as e:
                logger.warning("Failed to cleanup figures/ directory: %s", str(e))

if __name__ == "__main__":
    try:
        processor = DocumentProcessor(
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "http://192.168.190.101:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "multi-agent-rag-data"),
            prefix=os.getenv("MINIO_PREFIX", "document/"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            output_dir="scratch",
            store_images_on_disk=True
        )
        text_docs, table_docs, image_docs = asyncio.run(processor.load_and_parse_documents())
        logger.info("Text documents: %d", len(text_docs))
        logger.info("Table documents: %d", len(table_docs))
        logger.info("Image documents: %d", len(image_docs))
        for doc in text_docs[:2]:
            logger.info("Text content: %s, Metadata: %s", doc.page_content[:100], doc.metadata)
        for doc in table_docs[:2]:
            logger.info("Table content: %s, Metadata: %s", doc.page_content[:100], doc.metadata)
        for doc in image_docs[:2]:
            logger.info("Image content: %s, Metadata: %s", doc.page_content[:100], doc.metadata)
        processor.cleanup()
    except Exception as e:
        logger.error("Test failed: %s", str(e))
        raise