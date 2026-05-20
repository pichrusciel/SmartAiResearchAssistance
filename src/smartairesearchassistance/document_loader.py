from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from .structlogger import logger 

def load_file_content(file_path: Path) -> Optional[str]:
    """
    Identifies a file extension and extracts its complete text content 
    using the appropriate LangChain document loader.
    """

    extension = file_path.suffix.lower()

    logger.info("Importing file: ", filepath=file_path)
    try:
        if extension == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            # Extract and join content blocks if multi-document collection returned
            return "\n\n".join(doc.page_content for doc in documents)
        
        elif extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            # PyPDFLoader returns a Document per page; join them with spacing
            return "\n\n".join(doc.page_content for doc in documents)

        else:
            #print(f"Format not supported: '{extension}' for file '{file_path.name}'")
            logger.warning("Format not supported", extension=extension, file_path_name=file_path.name) 
            return None

    except Exception as e:
        #print(f"Error while loading file '{file_path.name}': {str(e)}")
        logger.error("Error while loading file: ", file_path_name=file_path.name, error_msg=str(e))
        return None