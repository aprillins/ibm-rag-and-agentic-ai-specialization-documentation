from crewai import Crew, Process, Agent, Task, LLM
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
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
  #tools=[SerperDevTool()]
)

research_task = Task(
  description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
  agent=research_agent,
  expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
)

crew = Crew(
    llm=llm,
    agents=[research_agent],
    tasks=[research_task],
    processes=Process.sequential,
    verbose=True
)
result = crew.kickoff(inputs={"topic": "Artificial Intelligence in Healthcare"})
print("Final Result:")
print(result.raw)