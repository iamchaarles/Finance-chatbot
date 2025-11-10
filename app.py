import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import time
from datetime import datetime

# Page Config
st.set_page_config(page_title="AI Finance Assistant", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""<style>
.main {background-color: #f0f2f6;}
.stButton>button {width: 100%;}
.stChatInputContainer textarea {
    border-color: #d3d3d3 !important;
    box-shadow: none !important;
}
.stChatInputContainer textarea:focus {
    border-color: #4CAF50 !important;
    box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25) !important;
}
</style>""", unsafe_allow_html=True)

# Session State Initialization
for key in ['messages', 'knowledge_base', 'last_request_time']:
    if key not in st.session_state:
        if key == 'messages':
            st.session_state[key] = []
        elif key == 'knowledge_base':
            st.session_state[key] = None
        else:
            st.session_state[key] = 0

# API Key
@st.cache_data(show_spinner=False)
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"].strip()
    except:
        st.error("⚠️ GROQ_API_KEY not configured in .streamlit/secrets.toml")
        st.info("Get your free API key at: https://console.groq.com/keys")
        return None

API_KEY = get_api_key()

# Knowledge Base
@st.cache_data
def get_knowledge_base():
    return {
        "sip": "SIP invests fixed amounts regularly in mutual funds. Best returns: Mid/Small-cap 15-20%. Start with ₹1,000/month.",
        "risk": "Conservative: Debt funds 5-7%. Moderate: Hybrid 8-12%. Aggressive: Equity 12-20%.",
        "emergency": "Emergency fund: 6-12 months expenses in liquid funds. Build before investing.",
        "tax": "80C: ELSS ₹1.5L (3-year lock), PPF 7.1%, NPS +₹50k. ELSS best for growth.",
        "stocks": "NIFTY 50: Top 50 companies. SENSEX: Top 30. Average 12-14% long-term.",
        "retirement": "Rule of 72: 72÷return = years to double. Invest 15-20% income. Target: 25-30x expenses.",
        "debt": "50-30-20 rule. Pay high-interest first. Credit card 36-42%, Personal 10-16%, Home 8-9%.",
        "gold": "Options: Physical, ETFs (0.5-1% cost), SGBs (2.5% interest). 9-10% returns.",
        "mutual fund": "Equity (high risk 12-20%), Debt (low risk 5-7%), Hybrid (balanced 8-12%).",
        "budget": "50-30-20: 50% needs, 30% wants, 20% savings. Track monthly.",
        "insurance": "Term: 10-15x income. Health: Min ₹5L. Keep insurance & investment separate."
    }

if not st.session_state.knowledge_base:
    st.session_state.knowledge_base = get_knowledge_base()

# Intent Detection
def detect_intent(query):
    """Detect user intent from query"""
    query_lower = query.lower().strip()
    
    # Greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon', 'namaste', 'hola']
    if any(greet in query_lower for greet in greetings) and len(query_lower.split()) <= 3:
        return 'greeting'
    
    # Gratitude
    thanks = ['thank', 'thanks', 'thank you', 'appreciate', 'helpful']
    if any(word in query_lower for word in thanks):
        return 'gratitude'
    
    # Casual/Small talk
    casual = ['how are you', 'what\'s up', 'whats up', 'who are you', 'your name']
    if any(phrase in query_lower for phrase in casual):
        return 'casual'
    
    # Specific finance topics
    if any(word in query_lower for word in ['sip', 'systematic investment', 'monthly investment']):
        return 'sip'
    if any(word in query_lower for word in ['tax', '80c', 'deduction', 'save tax']):
        return 'tax'
    if any(word in query_lower for word in ['stock', 'share', 'equity', 'nifty', 'sensex']):
        return 'stocks'
    if any(word in query_lower for word in ['retire', 'retirement', 'pension']):
        return 'retirement'
    if any(word in query_lower for word in ['risk', 'safe', 'conservative', 'aggressive']):
        return 'risk'
    if any(word in query_lower for word in ['emi', 'loan', 'debt', 'credit']):
        return 'debt'
    if any(word in query_lower for word in ['budget', 'save', 'saving', 'expense']):
        return 'budget'
    if any(word in query_lower for word in ['insurance', 'term', 'health insurance']):
        return 'insurance'
    if any(word in query_lower for word in ['mutual fund', 'mf', 'fund']):
        return 'mutual_fund'
    if any(word in query_lower for word in ['emergency', 'emergency fund', 'contingency']):
        return 'emergency'
    
    return 'general'

def get_quick_response(intent, query):
    """Generate quick responses for common intents"""
    responses = {
        'greeting': "Hey there! 👋 Great to see you! I'm Finny, your finance buddy. Whether you're looking to start investing, save on taxes, or just figure out where your money should go - I'm here to help! What's on your mind today?",
        'gratitude': "You're very welcome! 😊 I'm really glad I could help. Feel free to come back anytime you have questions about your finances. Wishing you great returns and smart money decisions! 💰",
        'casual': "I'm Finny, your friendly neighborhood finance advisor! 🤓 I specialize in Indian markets and love helping people make smart money moves. I'm doing great, thanks for asking! Now, let's talk about YOUR finances - what would you like to know?",
    }
    return responses.get(intent)

# GROQ API Response Function
def get_groq_response(prompt, intent, conversation_context=None):
    """Generate AI response using GROQ API with Llama 3.1"""
    if not API_KEY:
        return "⚠️ GROQ API key not configured. Please add it to .streamlit/secrets.toml"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # System prompts based on intent
    system_prompts = {
        'sip': "Focus on SIP benefits, how to start, recommended amounts, and best funds for beginners in India.",
        'tax': "Explain Indian tax-saving options like 80C, ELSS, PPF, NPS with clear comparisons.",
        'stocks': "Discuss Indian stock market basics, index funds vs individual stocks, and risk management.",
        'retirement': "Guide on retirement planning in India, corpus calculation, and investment strategies.",
        'risk': "Help assess risk tolerance and recommend appropriate investment products for Indian investors.",
        'debt': "Advise on loan management, EMI calculations, and debt reduction strategies in Indian context.",
        'budget': "Suggest budgeting frameworks like 50-30-20, expense tracking tips for Indian households.",
        'insurance': "Explain term insurance, health insurance in India, and why to separate investment from insurance.",
        'mutual_fund': "Compare Indian equity, debt, and hybrid funds with risk-return profiles.",
        'emergency': "Stress importance of emergency fund, ideal amount (6-12 months), and liquid fund options in India.",
        'general': "Understand the user's financial situation and provide personalized guidance for Indian markets."
    }
    context_hint = system_prompts.get(intent, system_prompts['general'])
    
    # System instruction
    system_instruction = f"""You are Finny, a warm and knowledgeable financial advisor specializing in Indian markets. You're like that smart friend who actually knows about money!

PERSONALITY TRAITS:
- Friendly and approachable - use conversational language, not jargon
- Empathetic - acknowledge the user's situation and concerns
- Encouraging - motivate them to take smart financial steps
- Clear - explain complex concepts simply with relatable examples
- Proactive - ask clarifying questions to give better advice
- Remember context - reference earlier parts of the conversation naturally

CONVERSATION GUIDELINES:
- NEVER start with "As a financial advisor..." - just chat naturally
- Use Indian context and examples (₹, Indian funds, tax rules, etc.)
- Show personality: occasional emojis, light humor, enthusiastic tone
- Ask follow-ups when helpful: "What's your current age?" "Do you have an emergency fund?"
- Keep it concise: 80-120 words unless detailed explanation needed
- When relevant, casually mention: "By the way, I have a SIP calculator in the sidebar if you want to run some numbers!"

TOPIC CONTEXT: {context_hint}

Respond naturally as Finny would - warm, helpful, and conversational!"""
    
    # Build messages array
    messages = [{"role": "system", "content": system_instruction}]
    
    # Add conversation context (last 6 messages)
    if conversation_context:
        for role, text in conversation_context:
            messages.append({
                "role": role if role == "user" else "assistant",
                "content": text
            })
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Request payload
    data = {
        "model": "llama-3.1-70b-versatile",  # Fast and high-quality
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 350,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        # Check for errors
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            return f"⚠️ GROQ API Error ({response.status_code}): {error_msg}"
        
        # Parse response
        result = response.json()
        
        if result.get('choices') and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "I'm having trouble generating a response right now. Please try rephrasing your question!"
    
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.RequestException as err:
        return f"⚠️ Connection error. Please check your internet and try again."
    except (KeyError, IndexError) as e:
        return "⚠️ Unexpected response format. Please try again."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def generate_response(query):
    """Main response generation function"""
    # Rate limiting
    if time.time() - st.session_state.last_request_time < 2:
        return "⚠️ Please wait 2 seconds between messages."
    st.session_state.last_request_time = time.time()
    
    intent = detect_intent(query)
    
    if API_KEY:
        # Check for quick responses
        quick_resp = get_quick_response(intent, query)
        if quick_resp:
            return quick_resp
        
        # Build conversation context (last 6 messages)
        conversation_context = [(msg['role'], msg['content']) for msg in st.session_state.messages[-6:]]
        
        return get_groq_response(query, intent, conversation_context=conversation_context)
    else:
        kb = st.session_state.knowledge_base
        context = kb.get(intent, "AI unavailable. Use sidebar tools for calculations.")
        return f"AI unavailable.\n\n{context}\n\n💡 Use sidebar tools or configure GROQ API key."

# Helper Functions
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1mo"):
    """Fetch stock data from yfinance"""
    try:
        for suffix in [".NS", ".BO"]:
            data = yf.Ticker(f"{symbol}{suffix}").history(period=period)
            if not data.empty:
                return data
    except:
        pass
    return None

@st.cache_data(ttl=300)
def get_index_data(symbol):
    """Get live index data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5d")
        if not data.empty:
            current = data['Close'].iloc[-1]
            previous = data['Close'].iloc[-2] if len(data) > 1 else current
            change_pct = ((current - previous) / previous) * 100
            return {
                'price': round(current, 2),
                'change_pct': round(change_pct, 2),
                'timestamp': data.index[-1].strftime('%d %b, %I:%M %p')
            }
    except:
        pass
    return None

def calc_sip(monthly, rate, years):
    """Calculate SIP returns"""
    months = years * 12
    r = rate / 1200
    fv = monthly * (((1 + r) ** months - 1) / r) * (1 + r) if r > 0 else monthly * months
    invested = monthly * months
    returns = fv - invested
    return {
        "fv": round(fv, 2),
        "invested": round(invested, 2),
        "returns": round(returns, 2),
        "pct": round(returns/invested*100, 2) if invested else 0
    }

def calc_emi(principal, rate, years):
    """Calculate EMI"""
    months = years * 12
    r = rate / 1200
    emi = principal * r * ((1 + r) ** months) / (((1 + r) ** months) - 1) if r > 0 else principal / months
    total = emi * months
    interest = total - principal
    return {
        "emi": round(emi, 2),
        "total": round(total, 2),
        "interest": round(interest, 2)
    }

def assess_risk(answers):
    """Assess risk profile"""
    score = sum(answers.values())
    if score <= 10:
        return "Conservative", "Debt funds/FDs (5-7%)", {"Debt": 70, "Equity": 20, "Gold": 10}
    elif score <= 20:
        return "Moderate", "Hybrid/Index funds (8-12%)", {"Debt": 40, "Equity": 45, "Gold": 15}
    return "Aggressive", "Equity funds (12-20%)", {"Debt": 20, "Equity": 70, "Other": 10}

# Main UI
st.title("💰 AI Finance Assistant")
st.markdown("*Your AI-powered finance advisor for Indian markets*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Tools")
    tool = st.selectbox("", ["💬 Chat", "📈 SIP Calculator", "🎯 Risk Assessment", "📊 Stocks", "💼 Budget", "🏦 EMI Calculator"])
    st.markdown("---")
    
    # API Status
    if API_KEY:
        st.success(" GROQ AI Enabled ✅")
    else:
        st.error(" AI Unavailable")
        st.info("Get free API key:\n[console.groq.com/keys](https://console.groq.com/keys)")
    
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()
    if col2.button("🔄 Refresh"):
        st.rerun()
    
    if st.session_state.messages:
        st.download_button(
            "📥 Export",
            json.dumps(st.session_state.messages, indent=2),
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )
    
    st.markdown("---")
    st.markdown("### 📊 Live Markets")
    
    nifty_data = get_index_data("^NSEI")
    sensex_data = get_index_data("^BSESN")
    
    if nifty_data:
        st.metric("NIFTY 50", f"{nifty_data['price']:,.0f}", f"{nifty_data['change_pct']:+.2f}%")
        st.caption(f"🕐 {nifty_data['timestamp']}")
    else:
        st.metric("NIFTY 50", "Fetching...", "")
    
    if sensex_data:
        st.metric("SENSEX", f"{sensex_data['price']:,.0f}", f"{sensex_data['change_pct']:+.2f}%")
        st.caption(f"🕐 {sensex_data['timestamp']}")
    else:
        st.metric("SENSEX", "Fetching...", "")

# Chat Tool
if tool == "💬 Chat":
    st.header("💬 Chat Assistant")
    
    # Initialize with greeting
    if not st.session_state.messages:
        initial_greeting = get_quick_response('greeting', 'hi')
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    
    st.info("💡 **Try asking:** 'I'm 25 and want to start investing, where should I begin?'")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle new input
    if prompt := st.chat_input("Chat with your finance advisor..."):
        if len(prompt.strip()) < 3:
            st.warning("Please enter a more detailed message.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# SIP Calculator
elif tool == "📈 SIP Calculator":
    st.header("📈 SIP Calculator")
    col1, col2 = st.columns(2)
    with col1:
        monthly = st.number_input("Monthly (₹)", 500, 1000000, 5000, 500)
        years = st.slider("Years", 1, 30, 10)
        rate = st.slider("Return %", 5.0, 25.0, 12.0, 0.5)
    
    with col2:
        if st.button("Calculate 🚀", type="primary"):
            r = calc_sip(monthly, rate, years)
            st.success("### Results")
            st.metric("Future Value", f"₹{r['fv']:,.0f}")
            st.metric("Invested", f"₹{r['invested']:,.0f}")
            st.metric("Returns", f"₹{r['returns']:,.0f} ({r['pct']:.1f}%)")
            st.bar_chart(pd.DataFrame({'Amount': [r['invested'], r['returns']]}, index=['Invested', 'Returns']))

# Risk Assessment
elif tool == "🎯 Risk Assessment":
    st.header("🎯 Risk Assessment")
    with st.form("risk"):
        q1 = st.radio("Investment goal?", ["Preservation (1)", "Income (2)", "Growth (3)", "Aggressive (4)"], index=2)
        q2 = st.radio("Time horizon?", ["<3yr (1)", "3-5yr (2)", "5-10yr (3)", ">10yr (4)"], index=2)
        q3 = st.radio("20% decline reaction?", ["Panic (1)", "Worried (2)", "Calm (3)", "Buy more (4)"], index=1)
        q4 = st.radio("Experience?", ["Beginner (1)", "Some (2)", "Experienced (3)", "Expert (4)"], index=1)
        q5 = st.radio("Income stability?", ["Unstable (1)", "Somewhat (2)", "Stable (3)", "Very stable (4)"], index=2)
        
        if st.form_submit_button("Assess 🎯", type="primary"):
            answers = {f'q{i+1}': int([q1,q2,q3,q4,q5][i].split('(')[1][0]) for i in range(5)}
            profile, rec, alloc = assess_risk(answers)
            st.success(f"### {profile} Profile")
            st.info(rec)
            for asset, pct in alloc.items():
                st.progress(pct/100, text=f"{asset}: {pct}%")

# Stock Lookup
elif tool == "📊 Stocks":
    st.header("📊 Stock Lookup")
    col1, col2 = st.columns([3, 1])
    symbol = col1.text_input("Symbol", "RELIANCE").upper().strip()
    period = col2.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
    
    quick = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "SBIN"]
    cols = st.columns(len(quick))
    for i, s in enumerate(quick):
        if cols[i].button(s, use_container_width=True):
            symbol = s
    
    if st.button("Fetch 🔍", type="primary") and symbol:
        with st.spinner("Fetching..."):
            data = get_stock_data(symbol, period)
            if data is not None:
                curr = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2] if len(data) > 1 else curr
                change = ((curr - prev) / prev) * 100
                last_update = data.index[-1].strftime('%d %b %Y, %I:%M %p')
                
                st.success(f"### {symbol}")
                st.caption(f"🕐 Last Updated: {last_update}")
                cols = st.columns(4)
                cols[0].metric("Price", f"₹{curr:.2f}", f"{change:+.2f}%")
                cols[1].metric("High", f"₹{data['High'].max():.2f}")
                cols[2].metric("Low", f"₹{data['Low'].min():.2f}")
                cols[3].metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                st.line_chart(data['Close'])
            else:
                st.error(f"No data for {symbol}")

# Budget Planner
elif tool == "💼 Budget":
    st.header("💼 Budget Planner (50-30-20)")
    income = st.number_input("Monthly Income (₹)", 10000, 10000000, 50000, 5000)
    
    if income > 0:
        needs, wants, savings = income * 0.50, income * 0.30, income * 0.20
        cols = st.columns(3)
        cols[0].metric("Needs (50%)", f"₹{needs:,.0f}")
        cols[1].metric("Wants (30%)", f"₹{wants:,.0f}")
        cols[2].metric("Savings (20%)", f"₹{savings:,.0f}")
        st.bar_chart(pd.DataFrame({'Amount': [needs, wants, savings]}, index=['Needs', 'Wants', 'Savings']))
        
        st.info(f"""**Recommendations:**
• Emergency Fund: ₹{income*6:,.0f} (6 months)
• Monthly SIP: ₹{savings*0.7:,.0f} (70% of savings)
• Insurance: ₹{needs*0.1:,.0f} (10% of needs)""")

# EMI Calculator
else:
    st.header("🏦 EMI Calculator")
    col1, col2 = st.columns(2)
    
    with col1:
        loan_type = st.selectbox("Type", ["Home (8.5%)", "Car (10.5%)", "Personal (14%)", "Education (9.5%)"])
        principal = st.number_input("Amount (₹)", 10000, 100000000, 2500000, 100000)
        rate = st.slider("Rate %", 5.0, 25.0, float(loan_type.split('(')[1].split('%')[0]))
        years = st.slider("Years", 1, 30, 20 if "Home" in loan_type else 5)
    
    with col2:
        if st.button("Calculate 💰", type="primary"):
            r = calc_emi(principal, rate, years)
            st.success("### EMI Details")
            st.markdown(f"## ₹{r['emi']:,.0f}")
            st.caption("Monthly EMI")
            cols = st.columns(2)
            cols[0].metric("Principal", f"₹{principal:,.0f}")
            cols[0].metric("Interest", f"₹{r['interest']:,.0f}")
            cols[1].metric("Total", f"₹{r['total']:,.0f}")
            cols[1].metric("Required Income", f"₹{r['emi']/0.4:,.0f}")
            st.bar_chart(pd.DataFrame({'Amount': [principal, r['interest']]}, index=['Principal', 'Interest']))

# Footer
st.markdown("---")
st.caption("⚠️ Educational purposes only. Consult SEBI registered advisor for personalized advice.")
st.caption("Powered by GROQ AI (Llama 3.1) 🚀")