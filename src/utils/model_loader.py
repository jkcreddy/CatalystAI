import os
import sys
import json
from dotenv import load_dotenv
from src.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import CatalystAIException


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS must be a JSON object.")
                self.api_keys = parsed
            except Exception as e:
                log.warning(f"Failed to parse API_KEYS env var: {e}")

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                value = os.getenv(key)
                if value:
                    self.api_keys[key] = value
                    log.info(f"Loaded {key} from individual env var.")

        # Final Check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error(f"Missing required API keys: {missing}")
            raise CatalystAIException(f"Missing required API keys", sys)
        
        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise CatalystAIException(f"API key '{key}' not found", sys)
        return val
    
class ModelLoader:
    """
    Load embedding models and LLMs based on configuration and API keys.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embedding(self):
        """
        Load and return the embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info(f"Loading embedding model: {model_name}") 
            #embedding = GoogleGenerativeAIEmbeddings(
            #    model=model_name,
            #    api_key=self.api_key_mgr.get("GOOGLE_API_KEY")
            #)
            embedding = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY")
            )
            log.info("Embedding model loaded successfully")
            return embedding
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            raise CatalystAIException("Failed to load embedding model", sys)
        
    
    def load_llm(self):
        """
        Load and return the LLM based on configuration.
        Supports Google Generative AI and Groq models.
        """
        llm_block = self.config["llm"]
        #provider_key = os.getenv("LLM_PROVIDER", "google")
        provider_key = os.getenv("LLM_PROVIDER", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embedding()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")