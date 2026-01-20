import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from PIL import Image
from dotenv import load_dotenv
import tools
load_dotenv()

# 2. Set up the Antropic model

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=1
)
tools_list = [tools.ocr_read_document]

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
agent = create_tool_calling_agent(llm, tools_list, prompt)



# 5. Set up the AgentExecutor to run the tool-enabled loop
agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=False)


ocr_output = tools.ocr_read_document.run("assets/invoice.png")

print("\n" + "─"*35 + " OCR OUTPUT " + "─"*33)
print(ocr_output[:600] + "..." if len(ocr_output) > 600 else ocr_output)

task = """
Please process the document at 'assets/invoice.png' using ocr
and extract the following information in JSON format:
- `All items listed 1 - 10`
"""

# Use .invoke() with a dictionary input for the agent_executor
response = agent_executor.invoke({"input": task})

# Display results side by side
print("="*80)
print(" " * 22 + "INVOICE PROCESSING RESULTS")
print("="*80)
print("\n" + "─"*35 + " OCR OUTPUT " + "─"*33)
print(ocr_output[:600] + "..." if len(ocr_output) > 600 else ocr_output)
print("\n" + "─"*35 + " LLM RESULT " + "─"*33)
print(response["output"])
print("="*80)