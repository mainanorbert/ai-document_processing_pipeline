from typing import Dict, Optional, Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from document_processor.services import ocr as tools

load_dotenv()


def run_ocr_extraction(image_path: str) -> str:
    """
    Extract plain text from an image using the configured OCR backend.
    """
    try:
        results = tools.ocr_read_document.invoke(image_path)
        return "\n".join(item["text"] for item in results if "text" in item)
    except Exception as exc:
        return f"OCR extraction failed: {exc}"


def normalize_llm_output(output: Any) -> str:
    """
    Convert agent output into a readable string.
    """
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        return "\n\n".join(
            item["text"] for item in output
            if isinstance(item, dict) and "text" in item
        )

    return str(output)


def run_llm_document_extraction(
    image_path: str,
    task_description: Optional[str] = None,
    model_name: str = "claude-3-haiku-20240307",
    temperature: float = 0.3,
) -> Dict[str, str]:
    """
    Run OCR + LLM agent to extract structured information from a document image.
    """
    instruction = task_description or "Extract all relevant information."

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=800,
    )

    tools_list = [tools.ocr_read_document]

    system_prompt = (
        "You are a document extraction assistant. "
        "Always use the OCR tool when given an image path. "
        "Return only the requested fields."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools_list,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        max_iterations=6,
        verbose=False,
    )

    task = f"""
Document path:
{image_path}

User request:
{instruction}
"""

    response = agent_executor.invoke({"input": task})
    llm_result = normalize_llm_output(response.get("output", ""))

    return {
        "success": True,
        "ocr_output": run_ocr_extraction(image_path),
        "llm_output": llm_result,
        "image_path": image_path,
    }