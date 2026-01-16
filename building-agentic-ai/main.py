from crewai import Crew, Process, Agent, Task, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()
topic = "Latest advancements in quantum computing"
search_result = search_tool.run(query=topic)
llm = LLM(model="gpt-4.1-mini", api_key=openai_api_key)

research_agent = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
  backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
  Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
  You excel at finding reliable sources and extracting valuable information efficiently.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[SerperDevTool()]
)

writer_agent = Agent(
  role='Tech Content Strategist',
  goal='Craft well-structured and engaging content based on research findings',
  backstory="""You are a skilled content strategist known for translating 
  complex topics into clear and compelling narratives. Your writing makes 
  information accessible and engaging for a wide audience.""",
  verbose=True,
  llm = llm,
  allow_delegation=True
)

research_task = Task(
  description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
  agent=research_agent,
  expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
)

writer_task = Task(
  description="Using the research report on {topic}, create an engaging article suitable for publication.",
  agent=writer_agent,
  expected_output="A well-structured article on {topic} based on the research findings."
)

crew = Crew(
    llm=llm,
    agents=[research_agent, writer_agent],
    tasks=[research_task, writer_task],
    processes=Process.sequential,
    verbose=True
)
result = crew.kickoff(inputs={"topic": topic})
print("Final Result:")
print(result.raw)