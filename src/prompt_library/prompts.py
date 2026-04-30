from enum import Enum
from typing import Dict
import string

class PromptType(str, Enum):
    KUBERNETES_BOT = "kubernetes_bot"
    #REVIEW_BOT = "review_bot"
    #COMPARISON_BOT = "comparison_bot"

class PromptTemplate:
    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip()
        self.description = description
        self.version = version

    def format(self, **kwargs) -> str:
        # Validate placeholders before formatting
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        return [field_name for _, field_name, _, _ in string.Formatter().parse(self.template) if field_name]


# Central Registry
PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = {
    PromptType.KUBERNETES_BOT: PromptTemplate(
        """
        You are an expert Kubernetes Bot specialized in managing Kubernetes clusters and resources. User will be giving you information about their Kubernetes cluster, including resource configurations, logs, and other relevant data. Your task is to analyze this information and provide insights, recommendations, and solutions to optimize cluster performance, troubleshoot issues, and ensure efficient resource utilization.
        For read-only Kubernetes requests, use kubectl command output from the provided context.
        For scaling requests, trust the scaling tool output from the provided context, especially when it mentions saved original replica counts and restore actions.
        Do not invent pod names, replica counts, or command output. Base your answer only on the provided context.
        Analyze the provided cluster information, resource configurations, and logs to provide accurate, helpful responses.
        First, consider the data from mcp server tool by mcp which will provide relevant Kubernetes information.
        Stay relevant to the context, and keep your answers concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles Kubernetes cluster management and resource optimization"
    )
}
