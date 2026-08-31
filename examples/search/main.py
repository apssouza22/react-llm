import os
from openai import OpenAI
from agents import people_search_agent, main_agent
from react.config import AgentConfig
from react.reactexecutor import ReActExecutor
from react.tools import search_wikipedia, perform_calculation, date_of_today, Tool

people_search_tool = Tool("People_search", people_search_agent, "To search for person information")
wikipedia_search_tool = Tool("WikipediaSearch", search_wikipedia, "To search for information on wikipedia")
calculator_tool = Tool("Calculator", perform_calculation, "To perform math calculations")
date_request_tool = Tool("Date_of_today", date_of_today, "To get the date of today")

open_ai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

if __name__ == "__main__":
    tools = [calculator_tool, date_request_tool, people_search_tool]
    main_agent.functions = tools
    people_search_agent.functions = [wikipedia_search_tool]

    agentConfig = AgentConfig()
    agentConfig.with_model_client(open_ai)
    agentConfig.with_token_limit(5000)
    agentConfig.with_max_interactions(10)
    react = ReActExecutor(agentConfig, main_agent)
    react.execute("What is the double of Linus Torvalds age?")
