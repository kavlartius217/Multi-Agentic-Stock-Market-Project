import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from crewai import Crew, Agent, Task, Process
from crewai_tools import SerperDevTool
from langchain_fmp_data import FMPDataTool
from langchain.chat_models import ChatOpenAI

# Set up API keys
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['SERPER_API_KEY'] = st.secrets['SERPER_API_KEY']
os.environ['FMP_API_KEY'] = st.secrets['FMP_API_KEY']
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

# Streamlit UI setup
st.set_page_config(page_title="AI Stock Analysis Crew", page_icon="📈")
st.title("📈 AI-Powered Stock Analysis Crew")

# Initialize tools
serper_tool = SerperDevTool()

class FinancialMarketDataTool(FMPDataTool):
    def _run(self, query: str) -> str:
        try:
            data = super()._run(query)
            return self._process_data(data)
        except Exception as e:
            return f"Error: {str(e)}"

    def _process_data(self, data):
        """Convert data to pandas DataFrame for better handling"""
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
        return data

fmp_tool = FinancialMarketDataTool()

# Initialize LLM
llm = ChatOpenAI(
    model="mixtral-8x7b-32768",
    temperature=0.3,
    openai_api_base="https://api.groq.com/openai/v1"
)

def setup_agents_tasks(stock_name):
    # Define Agents
    stock_analyst = Agent(
        role="Stock Data Analyst",
        goal=f"Analyze {stock_name} historical data and financial metrics",
        backstory="Expert in financial data analysis with deep market understanding",
        tools=[fmp_tool],
        verbose=True,
        llm=llm,
        memory=True
    )

    news_analyst = Agent(
        role="Financial News Researcher",
        goal=f"Find latest news and sentiment about {stock_name}",
        backstory="Experienced financial journalist with market sentiment analysis skills",
        tools=[serper_tool],
        verbose=True,
        llm=llm
    )

    viz_agent = Agent(
        role="Data Visualizer",
        goal="Create insightful visualizations from stock data",
        backstory="Data visualization expert specializing in financial markets",
        verbose=True,
        llm=llm
    )

    research_analyst = Agent(
        role="Senior Research Analyst",
        goal="Generate comprehensive research reports",
        backstory="Lead analyst with expertise in combining quantitative and qualitative data",
        verbose=True,
        llm=llm,
        memory=True
    )

    financial_advisor = Agent(
        role="Chief Financial Advisor",
        goal="Provide final investment recommendations",
        backstory="Seasoned Wall Street advisor with proven track record",
        verbose=True,
        llm=llm
    )

    # Define Tasks
    data_task = Task(
        description=f"""
        Analyze {stock_name} stock for the current year (2025):
        1. Retrieve last 30 days historical data
        2. Calculate key metrics (RSI, Moving Averages, Volatility)
        3. Identify significant trends and patterns
        """,
        expected_output="Structured report with data table and technical analysis",
        agent=stock_analyst,
        output_file="stock_data.md"
    )

    news_task = Task(
        description=f"""
        Gather latest news about {stock_name}:
        1. Search for recent announcements, earnings reports, and market news
        2. Analyze sentiment from news articles
        3. Identify potential market movers
        """,
        expected_output="News summary with sentiment analysis and source links",
        agent=news_analyst,
        output_file="news_report.md",
        async_execution=True
    )

    viz_task = Task(
        description="""
        Create visualizations from the stock data:
        1. Generate line chart for price movements
        2. Create volume analysis charts
        3. Produce technical indicator plots
        """,
        expected_output="Python code for Seaborn/Matplotlib visualizations",
        agent=viz_agent,
        context=[data_task],
        output_file="visualizations.py"
    )

    research_task = Task(
        description="""
        Combine data analysis and news insights:
        1. Create comprehensive research report
        2. Highlight key findings from both technical and fundamental analysis
        3. Include visualizations from data team
        """,
        expected_output="Formatted research report with charts and analysis",
        agent=research_analyst,
        context=[data_task, news_task, viz_task],
        output_file="full_report.md"
    )

    recommendation_task = Task(
        description="""
        Generate final investment recommendation:
        1. Consider all previous analysis
        2. Provide clear Buy/Hold/Sell recommendation
        3. Support conclusion with data-driven arguments
        """,
        expected_output="Executive summary with risk assessment and price targets",
        agent=financial_advisor,
        context=[research_task],
        output_file="recommendation.md"
    )

    return Crew(
        agents=[stock_analyst, news_analyst, viz_agent, research_analyst, financial_advisor],
        tasks=[data_task, news_task, viz_task, research_task, recommendation_task],
        process=Process.hierarchical,
        manager_llm=llm
    )

# Streamlit UI Components
with st.sidebar:
    st.header("Parameters")
    stock_name = st.text_input("Stock Ticker", "NVDA").upper()
    analysis_days = st.slider("Analysis Period (days)", 7, 90, 30)
    run_analysis = st.button("Start Analysis")

def display_results():
    """Display results from output files"""
    try:
        with open("stock_data.md") as f:
            st.subheader("Technical Analysis")
            st.markdown(f.read())
        
        with open("news_report.md") as f:
            st.subheader("Market News & Sentiment")
            st.markdown(f.read())
        
        st.subheader("Data Visualizations")
        viz_code = open("visualizations.py").read()
        exec(viz_code, globals(), locals())
        fig = plt.gcf()
        st.pyplot(fig)
        
        with open("full_report.md") as f:
            st.subheader("Full Research Report")
            st.markdown(f.read())
            
        with open("recommendation.md") as f:
            st.subheader("Investment Recommendation")
            st.markdown(f.read())
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

# Main execution flow
if run_analysis:
    with st.status("🚀 Initializing Analysis Crew...", expanded=True) as status:
        try:
            st.write("🔧 Setting up agents and tasks...")
            crew = setup_agents_tasks(stock_name)
            
            status.update(label="🧠 Running analysis pipeline...", state="running")
            result = crew.kickoff(inputs={'stock_name': stock_name})
            
            status.update(label="✅ Analysis complete!", state="complete")
            st.balloons()
            
        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.stop()

    display_results()

else:
    st.info("👈 Enter a stock ticker and configure analysis parameters to begin")
    st.image("https://i.imgur.com/6H3Bvvm.png", caption="Analysis Workflow")

# Style adjustments
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background: linear-gradient(45deg, #1a237e 30%, #0d47a1 90%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00c853 30%, #00e676 90%);
        color: white !important;
    }
    h1 {
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)
