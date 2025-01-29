import streamlit as st
from crewai import Crew, Process, Agent, Task
from crewai_tools import SerperDevTool, CodeInterpreterTool, BaseTool
from langchain_fmp_data import FMPDataTool
from pydantic import Field
import os
# import crewai_tools
# from langchain_community.tools.embed_file import EmbedFileTool
# from crewai.tools.embed_file import EmbedFileTool

# Set API keys (replace with your actual keys)
# Ensure these are set in your environment or Streamlit secrets

if "OPENAI_API_KEY" not in os.environ:
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

if "SERPER_API_KEY" not in os.environ:
    os.environ['SERPER_API_KEY'] = st.secrets["SERPER_API_KEY"]

if "FMP_API_KEY" not in os.environ:
    os.environ['FMP_API_KEY'] = st.secrets["FMP_API_KEY"]

# Tools
serper_tool = SerperDevTool()
fmp_tool = FMPDataTool()

class FinancialMarketDataTool(BaseTool):
    name: str = "Financial_Market_Data"
    description: str = (
        "Useful for retrieving financial market data. Use this tool to gather information"
        " about stock prices, financial metrics, company profiles, and more."
    )
    tool: fmp_tool = Field(default_factory=FMPDataTool)

    def _run(self, query: str) -> str:
        """Execute the financial market data query and return results."""
        try:
            return self.tool.run(query)
        except Exception as e:
            return f"Error fetching financial market data: {str(e)}"

# Function calling LLM (using gpt-4-turbo-preview as a placeholder, replace if needed)
from crewai import LLM
llm = LLM(model="gpt-4-turbo-preview")

# Agents
stock_analyst_agent = Agent(
    role="Stock Market Researcher",
    goal=(
        "All the data should be relative to the current year 2025"
        "Analyze a given stock by retrieving its historical data for the last 30 days. "
        "Represent this data in a well-structured tabular format. "
        "Additionally, provide detailed and insightful observations about trends, patterns, "
        "or anomalies within the data, such as price fluctuations, volume changes, or other "
        "noteworthy market indicators."
    ),
    backstory=(
        "You are an experienced stock market researcher with a deep understanding of financial markets, "
        "economic factors, and data analysis. Your expertise enables you to identify critical patterns "
        "and trends that can inform investment decisions."
    ),
    tools=[FinancialMarketDataTool()],
    memory=True,
    verbose=True,
    allow_delegation=True,
    function_calling_llm=llm,
)

news_analyst_agent = Agent(
    role="Stock News Analyst",
    goal=(
        "Search for and retrieve the latest stock-related news about a given company. "
        "Focus on identifying news that might have an impact on the company's stock price, such as "
        "earnings reports, management changes, mergers and acquisitions, regulatory developments, or "
        "industry trends. Provide a concise summary of the news in an easy-to-read format, highlighting "
        "key points and potential implications for the stock market."
    ),
    backstory=(
        "You are an expert in financial journalism and market analysis with a sharp eye for identifying "
        "news that can influence stock prices. Your insights help investors make informed decisions "
        "by understanding the impact of current events on financial markets."
    ),
    tools=[serper_tool],
    memory=False,
    verbose=True,
    allow_delegation=False,
    function_calling_llm=llm,
)

visualization_agent = Agent(
    role="As a data visualization expert, you are tasked with generating Seaborn commands based on the tabular data in the report generated by the 'stock_analyst_agent'.",
    goal="Your goal is to create visualizations that accurately represent the insights in the tabular data using the Seaborn library.",
    backstory="You have extensive experience in data visualization and proficiency in using Seaborn to create meaningful and informative plots for analyzing complex datasets.",
    memory=True,
    verbose=True
)

financial_advisor_agent = Agent(
    role="Analyze the data received from the 'stock_analyst_agent' and 'news_analyst_agent' to provide the user with informed stock recommendations. Assess market trends, stock performance, and news sentiment to determine whether the user should buy a stock, hold an already purchased stock, or sell it.",
    goal="Generate a comprehensive, data-driven financial report with clear investment recommendations.",
    backstory="You are a seasoned financial advisor specializing in stock market analysis. Your expertise lies in evaluating stock performance, market conditions, and financial news to offer precise and strategic investment advice.",
    memory=True,
    verbose=True
)

# Tasks
task1 = Task(
    description="Analyze the historical data of {stock_name} for the last 30 days and provide a well-structured report. All the data should be relative to the current year 2025",
    expected_output="A comprehensive report including: "
    "1. A tabular representation of the stock's historical data. "
    "2. Observations about trends, patterns, or anomalies within the data, such as price fluctuations, volume changes, or other noteworthy market indicators. "
    "3. Insights into potential investment opportunities or risks associated with the stock.",
    agent=stock_analyst_agent,
    tools=[FinancialMarketDataTool()],
    output_file="Stock_Analysis_Report_{stock_name}.md",
)

task2 = Task(
    description=(
        "Search and retrieve the latest impactful news about the stock of a company, "
        "{stock_name}. Focus on news that may influence the stock price, such as earnings reports, "
        "management changes, mergers, acquisitions, regulatory updates, or significant industry trends."
    ),
    expected_output=(
        "A detailed report in the following format:\n\n"
        "- **Company Name**: {stock_name}\n"
        "- **News Highlights**:\n"
        "  1. {Key Point 1}\n"
        "  2. {Key Point 2}\n"
        "- **Implications**: {Summary of the potential market impact}\n"
        "- **Sources**: {List of credible sources used}\n"
    ),
    agent=news_analyst_agent,
    tools=[serper_tool],
    async_execution=True,
    output_file="news.md",
)

task3 = Task(
    description="Using the report generated by the 'stock_analyst' agent, create Seaborn visualizations as specified in the report. First, construct a Pandas DataFrame from the tabular data, ensuring all arrays have equal lengths. The output should contain only the necessary Seaborn commands for visualization, without any additional scripting or code",
    expected_output="A Python file containing only Seaborn commands to visualize the data. The DataFrame must have equal-length arrays, and the visualizations should comprehensively represent the tabular data in all possible ways",
    agent=visualization_agent,
    output_file="visualizations.py",
    context=[task1],
)

task4 = Task(
    description="Analyze stock performance, market trends, and financial news to generate a comprehensive investment report. Provide clear recommendations on whether to buy, hold, or sell the stock based on the insights from 'stock_analyst_agent' and 'news_analyst_agent'.",
    expected_output="A well-structured financial report with data-driven investment recommendations.",
    agent=financial_advisor_agent,
    context=[task1, task2],
    output_file="overall_report.md",
)

# Streamlit App
st.set_page_config(
    page_title="Stock Analysis AI Crew",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("📈 Stock Analysis AI Crew")

    st.sidebar.header("Configuration")
    stock_name = st.sidebar.text_input("Enter Stock Name (e.g., Nvidia, Apple)", "Nvidia")

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing... Please wait."):
            crew = Crew(
                tasks=[task1, task2, task3, task4],
                agents=[
                    stock_analyst_agent,
                    news_analyst_agent,
                    visualization_agent,
                    financial_advisor_agent,
                ],
            )
            result = crew.kickoff({"stock_name": stock_name})

            st.success("Analysis Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Stock Analysis Report")
                try:
                    with open(f"Stock_Analysis_Report_{stock_name}.md", "r") as f:
                        stock_report = f.read()
                    st.markdown(stock_report)
                except FileNotFoundError:
                    st.error(f"Stock Analysis Report for {stock_name} not found.")

            with col2:
                st.header("News Report")
                try:
                    with open("news.md", "r") as f:
                        news_report = f.read()
                    st.markdown(news_report)
                except FileNotFoundError:
                    st.error("News Report not found.")

            with col3:
                st.header("Overall Report")
                try:
                    with open("overall_report.md", "r") as f:
                        overall_report = f.read()
                    st.markdown(overall_report)
                except FileNotFoundError:
                    st.error("Overall Report not found.")

            # Visualizations
            st.header("Visualizations")
            try:
                with open("visualizations.py", "r") as f:
                    viz_code = f.read()
                
                # Execute the visualization code
                exec(viz_code)

                # Display plots (replace with your actual plot display logic)
                # For example, if using matplotlib:
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # ... Seaborn plotting commands ...
                # st.pyplot(fig)

            except FileNotFoundError:
                st.error("Visualizations file not found.")
            except Exception as e:
                st.error(f"Error executing visualization code: {e}")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app uses a Crew of AI agents to analyze stock data, news, and generate investment recommendations."
    )

if __name__ == "__main__":
    main()
