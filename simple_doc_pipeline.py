import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import pytesseract
from langchain.tools import tool
from PIL import Image


from dotenv import load_dotenv

load_dotenv()

@tool
def ocr_read_document(image_path: str) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        return f"Error reading image: {e}"

# 2. Set up the Antropic model

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=1
)
tools = [ocr_read_document]

# 3. Create the Anthropic-compatible prompt
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

# 4. Create a proper tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. Set up the AgentExecutor to run the tool-enabled loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


