from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from document_processor.services import ocr as tools

load_dotenv()


def run_ocr_extraction(image_path: str) -> str:
    """
    Extract raw text from an image using the PaddleOCR tool.
    """
    try:
        ocr_result = tools.paddle_ocr_read_document.invoke(image_path)

        if (
            isinstance(ocr_result, list)
            and ocr_result
            and isinstance(ocr_result[0], dict)
        ):
            text_lines = [
                item.get("text", "")
                for item in ocr_result
                if "text" in item
            ]
            return "\n".join(text_lines)

        return str(ocr_result)

    except Exception as exc:
        return f"OCR extraction failed: {exc}"


def run_llm_document_extraction(
    image_path: str,
    task_description: Optional[str] = None,
    model_name: str = "claude-3-haiku-20240307",
    temperature: float = 0.3,
) -> Dict[str, str]:
    """
    Run OCR + LLM agent to extract structured data from a document image.
    """
    if task_description is None:
        task_description = f"""
Please process the document at '{image_path}' using OCR
and extract the information in a structured format. When reporting about extracted information,
clearly label each field and only include the requested fields without extraneous information.
"""

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=800,
    )

    tools_list = [tools.paddle_ocr_read_document]

    system_prompt = (
        "You are a helpful assistant designed to extract information "
        "from documents. You have access to an OCR tool that extracts "
        "raw text from images. Always report extracted results clearly "
        "and only include the requested fields without extraneous information."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools_list,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

    try:
        response = agent_executor.invoke(
            {"input": task_description}
        )
        llm_result = response.get(
            "output", "No output received"
        )
    except Exception as exc:
        llm_result = f"Agent execution failed: {exc}"

    ocr_text = run_ocr_extraction(image_path)

    return {
        "success": True,
        "ocr_output": ocr_text,
        "llm_output": llm_result,
        "image_path": image_path,
    }