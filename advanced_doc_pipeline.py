import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import tools

load_dotenv()


def run_ocr_extraction(image_path: str) -> str:
    """
    Extract raw text from an image using the PaddleOCR tool.
    
    Args:
        image_path: Path to the document image (default: 'assets/invoice.png')
    
    Returns:
        Extracted text as a string (or error message)
    """
    try:
        # Run the OCR tool directly
        ocr_result = tools.paddle_ocr_read_document.invoke(image_path)
        
        # If your tool returns structured data, convert to plain text
        if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], dict):
            # Assuming your tool returns list of dicts with 'text' key
            text_lines = [item.get('text', '') for item in ocr_result if 'text' in item]
            return "\n".join(text_lines)
        
        # If it already returns a string
        return str(ocr_result)
    
    except Exception as e:
        return f"OCR extraction failed: {str(e)}"


def run_llm_document_extraction(
    image_path: str,
    task_description: str = None,
    model_name: str = "claude-3-haiku-20240307",  # use a real, stable model name
    temperature: float = 0.3
) -> dict:
    """
    Set up Anthropic LLM + agent and extract structured information from a document.
    
    Args:
        image_path: Path to the image (passed to the task)
        task_description: Custom task prompt (defaults to invoice total extraction)
        model_name: Claude model identifier
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        Dict with keys: 'ocr_output', 'llm_output', 'success'
    """
    # Default task if none provided
    if task_description is None:
        task_description = f"""
Please process the document at '{image_path}' using OCR
and extract the following information in JSON format:
- total_amount_of_the_invoice (just the number, e.g. 123.45)
- currency (if detectable, e.g. USD, EUR)
- invoice_number (if present)
"""

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=800,
    )

    tools_list = [tools.paddle_ocr_read_document]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant designed to extract information from documents."
                "You have access to this tool: "
                "OCR tool to extract raw text from images"
                "Älways after successfully extracting data from images, report this way:'Here is your requested data..'. Only provide requested details in bullet form without additional explanation unless requested"
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools_list, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=6
    )

    try:
        response = agent_executor.invoke({"input": task_description})
        llm_result = response.get("output", "No output received")
    except Exception as e:
        llm_result = f"Agent execution failed: {str(e)}"

    # 7. Get raw OCR for display (optional but useful)
    ocr_text = run_ocr_extraction(image_path)

    return {
        "success": True,
        "ocr_output": ocr_text,
        "llm_output": llm_result,
        "image_path": image_path
    }

if __name__ == "__main__":
    image_path = "assets/invoice.png"
     
    print("Starting OCR extraction...")
    ocr_text = run_ocr_extraction(image_path)

    print("\n" + "─"*35 + " RAW OCR OUTPUT " + "─"*33)
    print(ocr_text[:800] + "..." if len(ocr_text) > 800 else ocr_text)

    print("\nStarting LLM extraction...")
    result = run_llm_document_extraction(
        image_path=image_path,
        temperature=0.2,  # lower for more consistent JSON
    )

    print("\n" + "─"*35 + " OCR OUTPUT " + "─"*33)
    print(result["ocr_output"][:800] + "..." if len(result["ocr_output"]) > 800 else result["ocr_output"])

    print("\n" + "─"*35 + " LLM EXTRACTION RESULT " + "─"*28)
    print(result["llm_output"])
    print("="*80)