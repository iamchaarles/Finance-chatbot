"""
AI Finance Chatbot - Main Application
Complete Streamlit application with LLM, RAG, and financial tools
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
# Removed unused sklearn import (caused import resolution error in some environments)
import chromadb
from chromadb.utils import embedding_functions
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stTextInput > div > div > input {background-color: white;}
    .chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex}
    .chat-message.user {background-color: #e3f2fd}
    .chat-message.assistant {background-color: #f1f8e9}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chroma_initialized' not in st.session_state:
    st.session_state.chroma_initialized = False

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the LLM model (using smaller model for CPU efficiency)"""
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Efficient for CPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

# ============================================================================
# CHROMADB INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_chroma():
    """Initialize ChromaDB with financial knowledge base"""
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="finance_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Sample financial knowledge base
    finance_docs = [
        {
            "text": "SIP (Systematic Investment Plan) is a disciplined investment approach where you invest fixed amounts regularly in mutual funds. Best SIP returns in 2024 came from Mid-cap and Small-cap funds with 15-20% returns.",
            "metadata": {"category": "investment", "topic": "sip"}
        },
        {
            "text": "Risk assessment involves evaluating risk tolerance, investment horizon, and financial goals. Conservative investors prefer debt funds (5-7% returns), moderate prefer hybrid (8-12%), aggressive prefer equity (12-20%).",
            "metadata": {"category": "risk", "topic": "assessment"}
        },
        {
            "text": "Emergency fund should cover 6-12 months of expenses. Keep it in liquid funds or savings accounts. Rule: Income - Savings = Expenses, not Income - Expenses = Savings.",
            "metadata": {"category": "planning", "topic": "emergency_fund"}
        },
        {
            "text": "Tax saving instruments: ELSS (80C, 1.5L limit), PPF (7.1% returns), NPS (additional 50k), Health insurance (80D). ELSS has 3-year lock-in and highest return potential.",
            "metadata": {"category": "tax", "topic": "saving"}
        },
        {
            "text": "Stock market indices: NIFTY 50 represents top 50 Indian companies, SENSEX represents top 30. Invest through index funds for diversification. Average long-term return: 12-14% annually.",
            "metadata": {"category": "stocks", "topic": "indices"}
        },
        {
            "text": "Retirement planning: Start early with Rule of 72 (72/return rate = years to double). Invest 15-20% of income. Use mix of EPF, NPS, mutual funds. Target corpus: 25-30x annual expenses.",
            "metadata": {"category": "retirement", "topic": "planning"}
        },
        {
            "text": "Debt management: Follow 50-30-20 rule (50% needs, 30% wants, 20% savings). Pay high-interest debt first. Credit card APR: 36-42%. Personal loan: 10-16%. Home loan: 8-9%.",
            "metadata": {"category": "debt", "topic": "management"}
        },
        {
            "text": "Gold investment options: Physical gold, Gold ETFs (low cost), Sovereign Gold Bonds (2.5% interest + price appreciation). Historical return: 9-10% annually.",
            "metadata": {"category": "gold", "topic": "investment"}
        }
    ]
    
    # Add documents if collection is empty
    if collection.count() == 0:
        texts = [doc["text"] for doc in finance_docs]
        metadatas = [doc["metadata"] for doc in finance_docs]
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
    
    return collection

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def retrieve_context(query, collection, n_results=3):
    """Retrieve relevant context from ChromaDB"""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
        return context
    except Exception as e:
        return ""

def get_stock_data(symbol, period="1mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")  # NSE stocks
        data = ticker.history(period=period)
        if data.empty:
            ticker = yf.Ticker(f"{symbol}.BO")  # BSE stocks
            data = ticker.history(period=period)
        return data
    except:
        return None

def calculate_sip_returns(monthly_investment, rate_of_return, years):
    """Calculate SIP returns using future value formula"""
    months = years * 12
    monthly_rate = rate_of_return / (12 * 100)
    
    if monthly_rate == 0:
        future_value = monthly_investment * months
    else:
        future_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    
    total_invested = monthly_investment * months
    returns = future_value - total_invested
    
    return {
        "future_value": round(future_value, 2),
        "total_invested": round(total_invested, 2),
        "returns": round(returns, 2),
        "return_percentage": round((returns / total_invested) * 100, 2)
    }

def assess_risk_profile(answers):
    """Assess user risk profile based on questionnaire"""
    score = sum(answers.values())
    
    if score <= 10:
        return "Conservative", "Low Risk - Focus on debt funds, FDs, bonds (5-7% returns)"
    elif score <= 20:
        return "Moderate", "Medium Risk - Balanced hybrid funds, index funds (8-12% returns)"
    else:
        return "Aggressive", "High Risk - Equity, mid-cap, small-cap funds (12-20% returns)"

def generate_response(query, context, tokenizer, model):
    """Generate response using LLM with context"""
    if tokenizer is None or model is None:
        return "Model not loaded. Using fallback response based on retrieved context."
    
    prompt = f"""<|system|>
You are a helpful financial advisor. Use the provided context to answer questions accurately and concisely.
</s>
<|user|>
Context: {context}

Question: {query}
</s>
<|assistant|>"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("<|assistant|>")[-1].strip()
        
        return response
    except Exception as e:
        return f"Based on the financial data: {context[:300]}..."

# ============================================================================
# MAIN UI
# ============================================================================

st.title("💰 AI Finance Assistant")
st.markdown("*Your personal finance advisor powered by AI*")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Tools & Features")
    
    tool = st.selectbox(
        "Select Tool",
        ["Chat Assistant", "SIP Calculator", "Risk Assessment", "Stock Lookup", "Budget Planner"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("NIFTY 50", "22,453", "+1.2%")
    st.metric("SENSEX", "74,085", "+0.8%")

# ============================================================================
# INITIALIZE CHROMADB
# ============================================================================

if not st.session_state.chroma_initialized:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.collection = initialize_chroma()
        st.session_state.chroma_initialized = True

# ============================================================================
# LOAD MODEL
# ============================================================================

tokenizer, model = load_model()

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

# ----------------------------------------------------------------------------
# CHAT ASSISTANT
# ----------------------------------------------------------------------------

if tool == "Chat Assistant":
    st.header("💬 Chat with AI Finance Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    # Chat input
    if prompt := st.chat_input("Ask about investments, SIPs, tax saving, budgeting..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve context
                context = retrieve_context(prompt, st.session_state.collection)
                
                # Generate response
                response = generate_response(prompt, context, tokenizer, model)
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# ----------------------------------------------------------------------------
# SIP CALCULATOR
# ----------------------------------------------------------------------------

elif tool == "SIP Calculator":
    st.header("📈 SIP Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_inv = st.number_input("Monthly Investment (₹)", min_value=500, value=5000, step=500)
        years = st.slider("Investment Period (Years)", 1, 30, 10)
        return_rate = st.slider("Expected Annual Return (%)", 5.0, 25.0, 12.0, 0.5)
    
    with col2:
        if st.button("Calculate Returns", type="primary"):
            results = calculate_sip_returns(monthly_inv, return_rate, years)
            
            st.success("### 💰 Investment Results")
            st.metric("Future Value", f"₹{results['future_value']:,.0f}")
            st.metric("Total Invested", f"₹{results['total_invested']:,.0f}")
            st.metric("Returns Earned", f"₹{results['returns']:,.0f}")
            st.metric("Return %", f"{results['return_percentage']:.2f}%")
            
            # Visualization
            data = pd.DataFrame({
                'Component': ['Total Invested', 'Returns'],
                'Amount': [results['total_invested'], results['returns']]
            })
            st.bar_chart(data.set_index('Component'))

# ----------------------------------------------------------------------------
# RISK ASSESSMENT
# ----------------------------------------------------------------------------

elif tool == "Risk Assessment":
    st.header("🎯 Risk Profile Assessment")
    
    st.write("Answer these questions to determine your investment risk profile:")
    
    q1 = st.radio("1. What is your investment goal?", 
                  ["Capital preservation (1)", "Steady income (2)", "Growth (3)", "Aggressive growth (4)"],
                  index=2)
    
    q2 = st.radio("2. Investment time horizon?",
                  ["< 3 years (1)", "3-5 years (2)", "5-10 years (3)", "> 10 years (4)"],
                  index=2)
    
    q3 = st.radio("3. How would you react to a 20% portfolio decline?",
                  ["Panic sell (1)", "Worried but hold (2)", "Stay calm (3)", "Buy more (4)"],
                  index=1)
    
    q4 = st.radio("4. Investment experience?",
                  ["Beginner (1)", "Some experience (2)", "Experienced (3)", "Expert (4)"],
                  index=1)
    
    q5 = st.radio("5. Income stability?",
                  ["Unstable (1)", "Somewhat stable (2)", "Stable (3)", "Very stable (4)"],
                  index=2)
    
    if st.button("Assess My Risk Profile", type="primary"):
        answers = {
            'q1': int(q1.split('(')[1][0]),
            'q2': int(q2.split('(')[1][0]),
            'q3': int(q3.split('(')[1][0]),
            'q4': int(q4.split('(')[1][0]),
            'q5': int(q5.split('(')[1][0])
        }
        
        profile, recommendation = assess_risk_profile(answers)
        
        st.success(f"### Your Risk Profile: {profile}")
        st.info(f"**Recommendation:** {recommendation}")
        
        if profile == "Conservative":
            st.write("**Suggested Allocation:**")
            st.write("- 70% Debt Funds / FDs")
            st.write("- 20% Large-cap Equity")
            st.write("- 10% Gold")
        elif profile == "Moderate":
            st.write("**Suggested Allocation:**")
            st.write("- 40% Debt Funds")
            st.write("- 45% Equity (Large + Mid cap)")
            st.write("- 15% Gold / International")
        else:
            st.write("**Suggested Allocation:**")
            st.write("- 20% Debt Funds")
            st.write("- 70% Equity (Mid + Small cap)")
            st.write("- 10% International / Sectoral")

# ----------------------------------------------------------------------------
# STOCK LOOKUP
# ----------------------------------------------------------------------------

elif tool == "Stock Lookup":
    st.header("📊 Stock Data Lookup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_symbol = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS, INFY)", "RELIANCE")
    
    with col2:
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
    
    if st.button("Fetch Stock Data", type="primary"):
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            data = get_stock_data(stock_symbol, period)
            
            if data is not None and not data.empty:
                st.success(f"### {stock_symbol} Stock Data")
                
                # Current price
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{current_price:.2f}", f"{change:+.2f}")
                col2.metric("Change %", f"{change_pct:+.2f}%")
                col3.metric("High", f"₹{data['High'].max():.2f}")
                col4.metric("Low", f"₹{data['Low'].min():.2f}")
                
                # Chart
                st.line_chart(data['Close'])
                
                # Data table
                st.dataframe(data.tail(10))
            else:
                st.error(f"Could not fetch data for {stock_symbol}. Please check the symbol.")

# ----------------------------------------------------------------------------
# BUDGET PLANNER
# ----------------------------------------------------------------------------

elif tool == "Budget Planner":
    st.header("💼 Budget Planner")
    
    st.write("Use the 50-30-20 rule for smart budgeting")
    
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    
    if monthly_income > 0:
        needs = monthly_income * 0.50
        wants = monthly_income * 0.30
        savings = monthly_income * 0.20
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💡 Needs (50%)", f"₹{needs:,.0f}")
            st.caption("Rent, food, utilities, EMIs")
        
        with col2:
            st.metric("🎉 Wants (30%)", f"₹{wants:,.0f}")
            st.caption("Entertainment, dining, shopping")
        
        with col3:
            st.metric("💰 Savings (20%)", f"₹{savings:,.0f}")
            st.caption("Investments, emergency fund")
        
        # Visualization
        budget_data = pd.DataFrame({
            'Category': ['Needs', 'Wants', 'Savings'],
            'Amount': [needs, wants, savings]
        })
        
        st.bar_chart(budget_data.set_index('Category'))
        
        st.info(f"""
        ### 📋 Budget Recommendations:
        - **Emergency Fund:** Build ₹{(monthly_income * 6):,.0f} (6 months expenses)
        - **Monthly SIP:** Consider ₹{(savings * 0.7):,.0f} in mutual funds
        - **Insurance:** Allocate ₹{(needs * 0.1):,.0f} for term + health insurance
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("*Disclaimer: This is an AI assistant for educational purposes. Consult a certified financial advisor for personalized advice.*")