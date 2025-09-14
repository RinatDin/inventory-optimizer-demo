"""
AI-Powered Inventory Optimization System
Full Working Prototype with Russian/English Support
Based on SCManagement1.0Alpha.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from openai import OpenAI
import json

# Page configuration
st.set_page_config(
    page_title="AI Inventory Optimizer | Оптимизатор Запасов",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


"""
AI-Powered Inventory Optimization System
Full Working Prototype with Russian/English Support
Based on SCManagement1.0Alpha.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from openai import OpenAI
import json

# Page configuration
st.set_page_config(
    page_title="AI Inventory Optimizer | Оптимизатор Запасов",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling and force light mode
st.markdown("""
    <style>
    /* Force light mode on all devices */
    :root {
        color-scheme: light !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Force light theme colors */
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Override dark mode text colors */
    .stMarkdown, .stText, p, span, div {
        color: #262730 !important;
    }
    
    /* Fix metric text colors */
    [data-testid="metric-container"] {
        background-color: #f0f2f6 !important;
        border: 2px solid #cbd5e0 !important;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    [data-testid="metric-container"] label {
        color: #262730 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #262730 !important;
    }
    
    /* Custom styling */
    .main {
        padding: 0rem 1rem;
        background-color: #ffffff !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .urgent-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6 !important;
        border-radius: 8px;
        color: #262730 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Fix dataframe colors */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stTable"] {
        background-color: #ffffff !important;
    }
    
    /* Fix select box and inputs */
    .stSelectbox label, .stTextInput label {
        color: #262730 !important;
    }
    
    /* Override Plotly dark theme */
    .js-plotly-plot .plotly {
        background-color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language toggle
col_lang1, col_lang2, col_lang3 = st.columns([8, 1, 1])
with col_lang2:
    if st.button("EN"):
        st.session_state.language = 'en'
        st.rerun()
with col_lang3:
    if st.button("RU"):
        st.session_state.language = 'ru'
        st.rerun()

# Translation dictionary
translations = {
    'en': {
        'title': '🤖 AI-Powered Inventory Optimizer',
        'total_skus': 'Total SKUs',
        'inventory_value': 'Inventory Value',
        'urgent_orders': '⚠️ Urgent Orders',
        'avg_turnover': 'Avg Turnover',
        'dashboard': '📊 Dashboard',
        'ai_assistant': '💬 AI Assistant',
        'analysis': '📈 Analysis',
        'what_if': '🔮 What-If Scenarios',
        'reorder': '📋 Smart Reorder'
    },
    'ru': {
        'title': '🤖 ИИ Оптимизатор Запасов',
        'total_skus': 'Всего SKU',
        'inventory_value': 'Стоимость Запасов',
        'urgent_orders': '⚠️ Срочные Заказы',
        'avg_turnover': 'Оборачиваемость',
        'dashboard': '📊 Панель',
        'ai_assistant': '💬 ИИ Ассистент',
        'analysis': '📈 Анализ',
        'what_if': '🔮 Сценарии',
        'reorder': '📋 Умный Заказ'
    }
}

t = translations[st.session_state.language]

# Load and process data
@st.cache_data
def load_inventory_data():
    """Load the inventory data - using dummy data for deployment"""
    # For deployment, always use dummy data
    # To use real Excel file locally, set USE_DUMMY_DATA = False
    USE_DUMMY_DATA = True  # Set to False to use real Excel file
    
    if not USE_DUMMY_DATA:
        try:
            # Read the Excel file
            df = pd.read_excel('SCManagment1.0Alpha.xlsx', sheet_name='Base', header=1)
            
            # Rename columns to English for easier processing
            column_mapping = {
                'Артикул': 'SKU',
                'Номенклатура': 'Product',
                'Характеристика': 'Characteristics',
                'Цвет': 'Color',
                'Себес За ШТ': 'Unit_Cost',
                'Выручка за ШТ': 'Revenue_Per_Unit',
                'Статус': 'Status',
                'ABC': 'ABC',
                'XYZ': 'XYZ',
                'Склад Шт': 'Stock_Qty',
                'Склад Руб': 'Stock_Value',
                'Транзит': 'Transit',
                'Оборачиваемость': 'Turnover',
                'Средние Продажи в Месяц': 'Avg_Monthly_Sales',
                'EOQ': 'EOQ',
                'MOQ': 'MOQ',
                'Лид Тайм': 'Lead_Time',
                'Safety Stock В ШТ': 'Safety_Stock',
                'Re-Order Point': 'Reorder_Point',
                'Срочные Заказы в ШТ': 'Urgent_Orders_Qty',
                'Срочные Заказы в Руб': 'Urgent_Orders_Value'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Clean data
            df = df[df['SKU'].notna()]
            numeric_columns = ['Stock_Qty', 'Stock_Value', 'Unit_Cost', 'Avg_Monthly_Sales', 
                              'EOQ', 'MOQ', 'Lead_Time', 'Safety_Stock', 'Reorder_Point', 
                              'Urgent_Orders_Qty', 'Urgent_Orders_Value', 'Turnover']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Calculate additional metrics
            df['Days_of_Stock'] = df.apply(
                lambda x: x['Stock_Qty'] / x['Avg_Monthly_Sales'] * 30 
                if x['Avg_Monthly_Sales'] > 0 else 0, axis=1
            )
            
            df['Stock_Status'] = df.apply(
                lambda x: 'Critical' if x['Stock_Qty'] < x['Safety_Stock'] 
                else 'Low' if x['Stock_Qty'] < x['Reorder_Point'] 
                else 'Optimal' if x['Stock_Qty'] < x['Reorder_Point'] * 2 
                else 'Excess', axis=1
            )
            
            df['Order_Urgency'] = df.apply(
                lambda x: 'Urgent' if x['Urgent_Orders_Qty'] > 0 
                else 'Soon' if x['Stock_Qty'] < x['Reorder_Point'] 
                else 'Normal', axis=1
            )
            
            return df
        except Exception as e:
            st.info("Using demo data for showcase")
            return create_sample_data()
    else:
        return create_sample_data()

def create_sample_data():
    """Create sample data if Excel file is not available"""
    np.random.seed(42)
    n_products = 50
    
    data = {
        'SKU': [f'SKU{i:04d}' for i in range(1, n_products + 1)],
        'Product': [f'Product {i}' for i in range(1, n_products + 1)],
        'ABC': np.random.choice(['A', 'B', 'C'], n_products, p=[0.2, 0.3, 0.5]),
        'XYZ': np.random.choice(['X', 'Y', 'Z'], n_products),
        'Stock_Qty': np.random.randint(0, 500, n_products),
        'Stock_Value': np.random.uniform(1000, 50000, n_products),
        'Unit_Cost': np.random.uniform(10, 500, n_products),
        'Avg_Monthly_Sales': np.random.uniform(10, 100, n_products),
        'Lead_Time': np.random.randint(7, 30, n_products),
        'Safety_Stock': np.random.randint(10, 100, n_products),
        'Reorder_Point': np.random.randint(20, 150, n_products),
        'EOQ': np.random.randint(50, 300, n_products),
        'MOQ': np.random.randint(25, 200, n_products),
        'Turnover': np.random.uniform(2, 12, n_products),
        'Urgent_Orders_Qty': np.random.choice([0, 0, 0, np.random.randint(10, 50)], n_products),
        'Urgent_Orders_Value': np.random.choice([0, 0, 0, np.random.uniform(1000, 10000)], n_products)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived fields
    df['Days_of_Stock'] = df['Stock_Qty'] / df['Avg_Monthly_Sales'] * 30
    df['Stock_Status'] = df.apply(
        lambda x: 'Critical' if x['Stock_Qty'] < x['Safety_Stock'] 
        else 'Low' if x['Stock_Qty'] < x['Reorder_Point'] 
        else 'Optimal' if x['Stock_Qty'] < x['Reorder_Point'] * 2 
        else 'Excess', axis=1
    )
    df['Order_Urgency'] = df.apply(
        lambda x: 'Urgent' if x['Urgent_Orders_Qty'] > 0 
        else 'Soon' if x['Stock_Qty'] < x['Reorder_Point'] 
        else 'Normal', axis=1
    )
    
    return df

# Load data
df = load_inventory_data()

# Header
st.title(t['title'])
st.markdown("---")

# Key Metrics Dashboard
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=t['total_skus'],
        value=len(df),
        delta=f"{len(df[df['ABC'] == 'A'])} A-class"
    )

with col2:
    total_value = df['Stock_Value'].sum()
    st.metric(
        label=t['inventory_value'],
        value=f"₽{total_value:,.0f}",
        delta=f"{(total_value / 1000000):.1f}M"
    )

with col3:
    urgent_count = len(df[df['Order_Urgency'] == 'Urgent'])
    urgent_value = df[df['Order_Urgency'] == 'Urgent']['Urgent_Orders_Value'].sum()
    st.metric(
        label=t['urgent_orders'],
        value=urgent_count,
        delta=f"₽{urgent_value:,.0f}" if urgent_value > 0 else "No urgent",
        delta_color="inverse" if urgent_count > 0 else "off"
    )

with col4:
    avg_turnover = df['Turnover'].mean()
    st.metric(
        label=t['avg_turnover'],
        value=f"{avg_turnover:.1f}x",
        delta="per year"
    )

with col5:
    critical_items = len(df[df['Stock_Status'] == 'Critical'])
    st.metric(
        label="⚠️ Critical Items",
        value=critical_items,
        delta="Need immediate action" if critical_items > 0 else "All good",
        delta_color="inverse" if critical_items > 0 else "off"
    )

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t['dashboard'], t['ai_assistant'], t['analysis'], t['what_if'], t['reorder']
])

# Tab 1: Dashboard
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock Status Overview
        fig_scatter = px.scatter(
            df, 
            x='Days_of_Stock', 
            y='Stock_Value',
            size='Stock_Qty',
            color='Stock_Status',
            hover_data=['SKU', 'Product', 'Stock_Qty', 'Reorder_Point'],
            title="Inventory Status Matrix",
            color_discrete_map={
                'Critical': '#FF4B4B',
                'Low': '#FFA500',
                'Optimal': '#00CC00',
                'Excess': '#4B4BFF'
            }
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # ABC-XYZ Matrix
        matrix_data = df.groupby(['ABC', 'XYZ']).size().reset_index(name='Count')
        fig_matrix = px.treemap(
            matrix_data,
            path=['ABC', 'XYZ'],
            values='Count',
            title="ABC-XYZ Classification"
        )
        fig_matrix.update_layout(height=400)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    # Critical Items Alert
    critical_df = df[df['Stock_Status'].isin(['Critical', 'Low'])].sort_values('Stock_Qty')
    
    if len(critical_df) > 0:
        st.markdown("### ⚠️ Items Requiring Attention")
        
        display_cols = ['SKU', 'Product', 'Stock_Qty', 'Reorder_Point', 'Safety_Stock', 
                       'Days_of_Stock', 'Stock_Status', 'Order_Urgency']
        
        # Style the dataframe
        def highlight_urgency(row):
            if row['Order_Urgency'] == 'Urgent':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Stock_Status'] == 'Critical':
                return ['background-color: #ffe6cc'] * len(row)
            elif row['Stock_Status'] == 'Low':
                return ['background-color: #ffffcc'] * len(row)
            return [''] * len(row)
        
        styled_df = critical_df[display_cols].style.apply(highlight_urgency, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Turnover Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Turnover by ABC
        turnover_abc = df.groupby('ABC')['Turnover'].mean().reset_index()
        fig_turnover = px.bar(
            turnover_abc,
            x='ABC',
            y='Turnover',
            title="Average Turnover by ABC Class",
            color='ABC',
            color_discrete_map={'A': '#00CC00', 'B': '#FFA500', 'C': '#FF4B4B'}
        )
        st.plotly_chart(fig_turnover, use_container_width=True)
    
    with col2:
        # Stock Value Distribution
        fig_pie = px.pie(
            df.groupby('ABC')['Stock_Value'].sum().reset_index(),
            values='Stock_Value',
            names='ABC',
            title="Inventory Value Distribution",
            color_discrete_map={'A': '#00CC00', 'B': '#FFA500', 'C': '#FF4B4B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 2: AI Assistant
with tab2:
    st.markdown("### 🤖 Intelligent Inventory Assistant")
    
    # Initialize OpenAI client
    try:
        # Try to get API key from secrets or environment
        api_key = st.secrets.get("OPENAI_API_KEY", None) or st.text_input("Enter OpenAI API Key:", type="password")
        
        if api_key:
            client = OpenAI(api_key=api_key)
            
            # Prepare context for AI
            inventory_context = f"""
            You are an expert inventory optimization assistant. Here's the current inventory status:
            
            Total SKUs: {len(df)}
            Total Inventory Value: ₽{df['Stock_Value'].sum():,.0f}
            Critical Items: {len(df[df['Stock_Status'] == 'Critical'])}
            Low Stock Items: {len(df[df['Stock_Status'] == 'Low'])}
            Average Turnover: {df['Turnover'].mean():.1f}x per year
            
            Top 5 Critical Items:
            {df[df['Stock_Status'] == 'Critical'][['SKU', 'Product', 'Stock_Qty', 'Days_of_Stock']].head().to_string()}
            
            ABC Distribution:
            - A items: {len(df[df['ABC'] == 'A'])} ({df[df['ABC'] == 'A']['Stock_Value'].sum()/df['Stock_Value'].sum()*100:.1f}% of value)
            - B items: {len(df[df['ABC'] == 'B'])} ({df[df['ABC'] == 'B']['Stock_Value'].sum()/df['Stock_Value'].sum()*100:.1f}% of value)
            - C items: {len(df[df['ABC'] == 'C'])} ({df[df['ABC'] == 'C']['Stock_Value'].sum()/df['Stock_Value'].sum()*100:.1f}% of value)
            
            You can answer questions about inventory optimization, suggest reorder strategies, 
            identify risks, and provide actionable recommendations.
            """
            
            if len(st.session_state.messages) == 0:
                st.session_state.messages = [{"role": "system", "content": inventory_context}]
            
            # Display chat history
            for message in st.session_state.messages[1:]:  # Skip system message
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about your inventory..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        # Add current data context to the prompt
                        messages_with_context = st.session_state.messages + [
                            {"role": "system", "content": f"Current data sample:\n{df.head(10).to_string()}"}
                        ]
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages_with_context,
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        answer = response.choices[0].message.content
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Quick action buttons
            st.markdown("#### Quick Actions:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Analyze Critical Items"):
                    prompt = "What are the most critical inventory issues right now and what should I do about them?"
                    with st.spinner("Analyzing critical items..."):
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": inventory_context},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        st.info(response.choices[0].message.content)
            
            with col2:
                if st.button("💡 Optimization Suggestions"):
                    prompt = "Provide 3 specific recommendations to optimize inventory turnover and reduce holding costs."
                    with st.spinner("Generating optimization suggestions..."):
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": inventory_context},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        st.success(response.choices[0].message.content)
            
            with col3:
                if st.button("⚠️ Risk Assessment"):
                    prompt = "Identify the top 3 inventory risks and their potential impact on operations."
                    with st.spinner("Assessing risks..."):
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": inventory_context},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        st.warning(response.choices[0].message.content)
        else:
            st.info("Please enter your OpenAI API key to enable AI features.")
    
    except Exception as e:
        st.error(f"AI Assistant Error: {str(e)}")
        st.info("The AI assistant requires an OpenAI API key. You can still use other features.")

# Tab 3: Analysis
with tab3:
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["ABC Analysis", "Turnover Analysis", "Stock Coverage", "Seasonal Patterns"]
    )
    
    if analysis_type == "ABC Analysis":
        # ABC Analysis with Pareto
        # Filter out any rows with null Stock_Value
        df_abc = df[df['Stock_Value'].notna() & (df['Stock_Value'] > 0)].copy()
        
        if len(df_abc) > 0:
            df_abc = df_abc.sort_values('Stock_Value', ascending=False)
            df_abc['Cumulative_Value'] = df_abc['Stock_Value'].cumsum()
            df_abc['Cumulative_Pct'] = df_abc['Cumulative_Value'] / df_abc['Stock_Value'].sum() * 100
            df_abc['Item_Pct'] = (np.arange(1, len(df_abc) + 1) / len(df_abc)) * 100
            
            fig_pareto = go.Figure()
            
            # Bar chart
            fig_pareto.add_trace(go.Bar(
                x=list(range(len(df_abc))),
                y=df_abc['Stock_Value'].tolist(),
                name='Stock Value',
                marker_color='lightblue'
            ))
            
            # Line chart
            fig_pareto.add_trace(go.Scatter(
                x=list(range(len(df_abc))),
                y=df_abc['Cumulative_Pct'].tolist(),
                name='Cumulative %',
                yaxis='y2',
                line=dict(color='red', width=2)
            ))
            
            # Add 80% reference line
            fig_pareto.add_trace(go.Scatter(
                x=[0, len(df_abc)-1],
                y=[80, 80],
                name='80% Line',
                yaxis='y2',
                line=dict(color='green', width=1, dash='dash'),
                showlegend=True
            ))
            
            fig_pareto.update_layout(
                title='ABC Analysis - Pareto Chart',
                xaxis=dict(title='Products (sorted by value)'),
                yaxis=dict(title='Stock Value (₽)'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
                height=500
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.warning("No valid data available for ABC Analysis")
        
        # ABC Summary
        col1, col2, col3 = st.columns(3)
        
        for col, abc_class in zip([col1, col2, col3], ['A', 'B', 'C']):
            with col:
                class_df = df[df['ABC'] == abc_class]
                st.markdown(f"### Class {abc_class}")
                st.metric("Items", len(class_df))
                st.metric("Value", f"₽{class_df['Stock_Value'].sum():,.0f}")
                st.metric("% of Total", f"{class_df['Stock_Value'].sum()/df['Stock_Value'].sum()*100:.1f}%")
    
    elif analysis_type == "Turnover Analysis":
        # Turnover analysis
        fig_turnover_dist = px.histogram(
            df,
            x='Turnover',
            nbins=20,
            title='Turnover Distribution',
            labels={'Turnover': 'Turnover Rate (times/year)', 'count': 'Number of Products'}
        )
        st.plotly_chart(fig_turnover_dist, use_container_width=True)
        
        # Slow movers
        slow_movers = df[df['Turnover'] < 2].sort_values('Stock_Value', ascending=False)
        st.markdown("### 🐌 Slow Moving Items (Turnover < 2)")
        st.dataframe(
            slow_movers[['SKU', 'Product', 'Stock_Value', 'Turnover', 'Days_of_Stock']].head(10),
            use_container_width=True
        )
    
    elif analysis_type == "Stock Coverage":
        # Days of stock analysis
        fig_coverage = px.box(
            df,
            x='ABC',
            y='Days_of_Stock',
            color='ABC',
            title='Stock Coverage by ABC Class',
            color_discrete_map={'A': '#00CC00', 'B': '#FFA500', 'C': '#FF4B4B'}
        )
        st.plotly_chart(fig_coverage, use_container_width=True)
        
        # Coverage categories
        coverage_bins = [0, 15, 30, 60, 90, float('inf')]
        coverage_labels = ['< 15 days', '15-30 days', '30-60 days', '60-90 days', '> 90 days']
        df['Coverage_Category'] = pd.cut(df['Days_of_Stock'], bins=coverage_bins, labels=coverage_labels)
        
        coverage_summary = df.groupby('Coverage_Category')['Stock_Value'].agg(['count', 'sum']).reset_index()
        coverage_summary.columns = ['Coverage', 'Count', 'Value']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_cov_count = px.pie(
                coverage_summary,
                values='Count',
                names='Coverage',
                title='Items by Stock Coverage'
            )
            st.plotly_chart(fig_cov_count, use_container_width=True)
        
        with col2:
            fig_cov_value = px.pie(
                coverage_summary,
                values='Value',
                names='Coverage',
                title='Value by Stock Coverage'
            )
            st.plotly_chart(fig_cov_value, use_container_width=True)
    
    elif analysis_type == "Seasonal Patterns":
        # Simulate seasonal data for demonstration
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create seasonal pattern for top products
        top_products = df.nlargest(5, 'Stock_Value')
        
        seasonal_data = []
        for _, product in top_products.iterrows():
            base_sales = product['Avg_Monthly_Sales']
            for i, month in enumerate(months):
                # Create seasonal variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 12)
                sales = base_sales * seasonal_factor * np.random.uniform(0.8, 1.2)
                seasonal_data.append({
                    'Product': product['Product'][:20],  # Truncate long names
                    'Month': month,
                    'Sales': sales
                })
        
        seasonal_df = pd.DataFrame(seasonal_data)
        
        fig_seasonal = px.line(
            seasonal_df,
            x='Month',
            y='Sales',
            color='Product',
            title='Seasonal Sales Patterns - Top 5 Products',
            markers=True
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

# Tab 4: What-If Scenarios
with tab4:
    st.markdown("### 🔮 What-If Scenario Analysis")
    st.markdown("Simulate different scenarios to understand inventory impact")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        demand_change = st.slider(
            "Demand Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Simulate demand increase or decrease"
        )
        
        lead_time_change = st.slider(
            "Lead Time Change (days)",
            min_value=-14,
            max_value=30,
            value=0,
            step=1,
            help="Simulate supplier lead time changes"
        )
        
        service_level = st.select_slider(
            "Target Service Level",
            options=[0.90, 0.95, 0.97, 0.99],
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%"
        )
        
        cost_change = st.slider(
            "Cost Change (%)",
            min_value=-20,
            max_value=50,
            value=0,
            step=5,
            help="Simulate cost inflation/deflation"
        )
        
        if st.button("🚀 Run Scenario", type="primary"):
            with st.spinner("Running scenario analysis..."):
                # Simulate scenario
                scenario_df = df.copy()
                
                # Apply changes
                scenario_df['Avg_Monthly_Sales'] *= (1 + demand_change/100)
                scenario_df['Lead_Time'] += lead_time_change
                scenario_df['Unit_Cost'] *= (1 + cost_change/100)
                
                # Recalculate metrics
                from scipy import stats
                z_score = stats.norm.ppf(service_level)
                
                # Simplified safety stock calculation
                scenario_df['Safety_Stock'] = z_score * np.sqrt(scenario_df['Lead_Time']) * scenario_df['Avg_Monthly_Sales'] * 0.3
                scenario_df['Reorder_Point'] = scenario_df['Lead_Time'] * scenario_df['Avg_Monthly_Sales']/30 + scenario_df['Safety_Stock']
                scenario_df['Stock_Value'] = scenario_df['Stock_Qty'] * scenario_df['Unit_Cost']
                
                # Calculate new status
                scenario_df['New_Status'] = scenario_df.apply(
                    lambda x: 'Critical' if x['Stock_Qty'] < x['Safety_Stock']
                    else 'Low' if x['Stock_Qty'] < x['Reorder_Point']
                    else 'Optimal', axis=1
                )
                
                st.session_state.scenario_results = scenario_df
                time.sleep(1)  # Simulate processing
                st.success("✅ Scenario analysis complete!")
                st.rerun()
    
    with col2:
        if st.session_state.scenario_results is not None:
            st.markdown("#### Scenario Impact Analysis")
            
            scenario_df = st.session_state.scenario_results
            
            # Compare metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                original_value = df['Stock_Value'].sum()
                new_value = scenario_df['Stock_Value'].sum()
                delta_value = ((new_value - original_value) / original_value * 100)
                st.metric(
                    "Total Inventory Value",
                    f"₽{new_value:,.0f}",
                    f"{delta_value:+.1f}%"
                )
            
            with col_m2:
                original_critical = len(df[df['Stock_Status'] == 'Critical'])
                new_critical = len(scenario_df[scenario_df['New_Status'] == 'Critical'])
                st.metric(
                    "Critical Items",
                    new_critical,
                    f"{new_critical - original_critical:+d}"
                )
            
            with col_m3:
                avg_safety = scenario_df['Safety_Stock'].mean()
                original_safety = df['Safety_Stock'].mean()
                st.metric(
                    "Avg Safety Stock",
                    f"{avg_safety:.0f}",
                    f"{((avg_safety - original_safety)/original_safety*100):+.1f}%"
                )
            
            with col_m4:
                total_reorder = len(scenario_df[scenario_df['Stock_Qty'] < scenario_df['Reorder_Point']])
                original_reorder = len(df[df['Stock_Qty'] < df['Reorder_Point']])
                st.metric(
                    "Items to Reorder",
                    total_reorder,
                    f"{total_reorder - original_reorder:+d}"
                )
            
            # Status comparison chart
            status_comparison = pd.DataFrame({
                'Status': ['Critical', 'Low', 'Optimal'],
                'Current': [
                    len(df[df['Stock_Status'] == 'Critical']),
                    len(df[df['Stock_Status'] == 'Low']),
                    len(df[df['Stock_Status'] == 'Optimal'])
                ],
                'Scenario': [
                    len(scenario_df[scenario_df['New_Status'] == 'Critical']),
                    len(scenario_df[scenario_df['New_Status'] == 'Low']),
                    len(scenario_df[scenario_df['New_Status'] == 'Optimal'])
                ]
            })
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(name='Current', x=status_comparison['Status'], y=status_comparison['Current']))
            fig_comparison.add_trace(go.Bar(name='Scenario', x=status_comparison['Status'], y=status_comparison['Scenario']))
            fig_comparison.update_layout(
                title='Stock Status Comparison',
                barmode='group',
                height=350
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Most impacted items
            scenario_df['Impact_Score'] = abs(scenario_df['Reorder_Point'] - df['Reorder_Point'])
            most_impacted = scenario_df.nlargest(10, 'Impact_Score')[['SKU', 'Product', 'Stock_Qty', 'Reorder_Point', 'New_Status']]
            
            st.markdown("##### Most Impacted Items")
            st.dataframe(most_impacted, use_container_width=True, height=200)
        else:
            st.info("👈 Adjust parameters and click 'Run Scenario' to see the impact")
            
            # Show example scenario suggestions
            st.markdown("#### 💡 Try These Scenarios:")
            
            scenarios = [
                {"name": "📈 High Season", "desc": "Demand +30%, Lead time +7 days", "icon": "🎄"},
                {"name": "📉 Recession", "desc": "Demand -20%, Cost +10%", "icon": "📊"},
                {"name": "🚢 Supply Chain Crisis", "desc": "Lead time +21 days, Cost +25%", "icon": "⚠️"},
                {"name": "🎯 Premium Service", "desc": "Service level 99%, normal conditions", "icon": "⭐"}
            ]
            
            for scenario in scenarios:
                with st.expander(f"{scenario['icon']} {scenario['name']}"):
                    st.write(scenario['desc'])
                    st.write("See how your inventory would respond to this scenario")

# Tab 5: Smart Reorder
with tab5:
    st.markdown("### 📋 AI-Powered Reorder Recommendations")
    
    # Filter for items needing reorder
    reorder_needed = df[
        (df['Stock_Qty'] < df['Reorder_Point']) | 
        (df['Order_Urgency'] == 'Urgent')
    ].copy()
    
    if len(reorder_needed) > 0:
        # Calculate reorder details
        reorder_needed['Suggested_Order_Qty'] = reorder_needed.apply(
            lambda x: max(x['EOQ'], x['MOQ']) if x['MOQ'] > 0 else x['EOQ'], axis=1
        )
        reorder_needed['Order_Value'] = reorder_needed['Suggested_Order_Qty'] * reorder_needed['Unit_Cost']
        
        # Calculate priority score with proper vectorized operation
        reorder_needed['Priority_Score'] = (
            (1 - reorder_needed['Stock_Qty'] / reorder_needed['Reorder_Point'].clip(lower=1)) * 
            reorder_needed['ABC'].map({'A': 1, 'B': 0.5, 'C': 0.2}).fillna(0.1)
        )
        
        # Sort by priority
        reorder_needed = reorder_needed.sort_values('Priority_Score', ascending=False)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Items to Reorder", len(reorder_needed))
        with col2:
            st.metric("Total Order Value", f"₽{reorder_needed['Order_Value'].sum():,.0f}")
        with col3:
            st.metric("Urgent Orders", len(reorder_needed[reorder_needed['Order_Urgency'] == 'Urgent']))
        with col4:
            st.metric("Critical Items", len(reorder_needed[reorder_needed['Stock_Status'] == 'Critical']))
        
        # Reorder strategy selector
        strategy = st.radio(
            "Select Reorder Strategy:",
            ["Optimal (EOQ-based)", "Conservative (Minimum)", "Aggressive (Maximum)", "Custom"],
            horizontal=True
        )
        
        if strategy == "Custom":
            custom_multiplier = st.slider("Order Quantity Multiplier", 0.5, 2.0, 1.0, 0.1)
            reorder_needed['Suggested_Order_Qty'] *= custom_multiplier
            reorder_needed['Order_Value'] = reorder_needed['Suggested_Order_Qty'] * reorder_needed['Unit_Cost']
        elif strategy == "Conservative":
            reorder_needed['Suggested_Order_Qty'] = reorder_needed['MOQ']
            reorder_needed['Order_Value'] = reorder_needed['Suggested_Order_Qty'] * reorder_needed['Unit_Cost']
        elif strategy == "Aggressive":
            reorder_needed['Suggested_Order_Qty'] = reorder_needed[['EOQ', 'MOQ']].max(axis=1) * 1.5
            reorder_needed['Order_Value'] = reorder_needed['Suggested_Order_Qty'] * reorder_needed['Unit_Cost']
        
        # Generate purchase orders
        if st.button("🤖 Generate AI Purchase Order Recommendations", type="primary"):
            with st.spinner("AI is analyzing and creating optimal purchase orders..."):
                time.sleep(2)  # Simulate AI processing
                
                # Group by characteristics for smart ordering
                st.success("✅ AI Purchase Order Recommendations Generated!")
                
                # Priority 1: Urgent Orders
                urgent_orders = reorder_needed[reorder_needed['Order_Urgency'] == 'Urgent']
                if len(urgent_orders) > 0:
                    st.markdown("#### 🔴 Priority 1: URGENT ORDERS (Immediate Action Required)")
                    st.dataframe(
                        urgent_orders[['SKU', 'Product', 'Stock_Qty', 'Suggested_Order_Qty', 'Order_Value']].head(10),
                        use_container_width=True
                    )
                    st.warning(f"⚠️ Total Urgent Order Value: ₽{urgent_orders['Order_Value'].sum():,.0f}")
                
                # Priority 2: Critical A-Class
                critical_a = reorder_needed[
                    (reorder_needed['ABC'] == 'A') & 
                    (reorder_needed['Stock_Status'] == 'Critical')
                ]
                if len(critical_a) > 0:
                    st.markdown("#### 🟡 Priority 2: Critical A-Class Items")
                    st.dataframe(
                        critical_a[['SKU', 'Product', 'Stock_Qty', 'Suggested_Order_Qty', 'Order_Value']].head(10),
                        use_container_width=True
                    )
                
                # Priority 3: Regular Reorders
                regular = reorder_needed[
                    (reorder_needed['Order_Urgency'] != 'Urgent') & 
                    ~((reorder_needed['ABC'] == 'A') & (reorder_needed['Stock_Status'] == 'Critical'))
                ]
                if len(regular) > 0:
                    st.markdown("#### 🟢 Priority 3: Regular Reorders")
                    st.dataframe(
                        regular[['SKU', 'Product', 'Stock_Qty', 'Suggested_Order_Qty', 'Order_Value']].head(10),
                        use_container_width=True
                    )
                
                # Download button for orders
                csv = reorder_needed[['SKU', 'Product', 'Suggested_Order_Qty', 'Order_Value']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Purchase Orders (CSV)",
                    data=csv,
                    file_name=f"purchase_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Visual order summary
        st.markdown("#### 📊 Order Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Orders by ABC class
            fig_abc_orders = px.pie(
                reorder_needed.groupby('ABC')['Order_Value'].sum().reset_index(),
                values='Order_Value',
                names='ABC',
                title='Order Value by ABC Class',
                color_discrete_map={'A': '#00CC00', 'B': '#FFA500', 'C': '#FF4B4B'}
            )
            st.plotly_chart(fig_abc_orders, use_container_width=True)
        
        with col2:
            # Top suppliers or categories
            fig_top_orders = px.bar(
                reorder_needed.nlargest(10, 'Order_Value'),
                x='Order_Value',
                y='Product',
                orientation='h',
                title='Top 10 Orders by Value',
                labels={'Order_Value': 'Order Value (₽)', 'Product': ''}
            )
            fig_top_orders.update_layout(height=400)
            st.plotly_chart(fig_top_orders, use_container_width=True)
        
        # AI Insights
        st.markdown("#### 🧠 AI Insights & Recommendations")
        
        insights = f"""
        Based on the current analysis:
        
        1. **Immediate Action Required**: {len(reorder_needed[reorder_needed['Order_Urgency'] == 'Urgent'])} items need urgent reordering to prevent stockouts.
        
        2. **Budget Impact**: Total reorder value of ₽{reorder_needed['Order_Value'].sum():,.0f} required, with ₽{reorder_needed[reorder_needed['ABC'] == 'A']['Order_Value'].sum():,.0f} for A-class items.
        
        3. **Risk Mitigation**: {len(reorder_needed[reorder_needed['Stock_Status'] == 'Critical'])} critical items should be prioritized to maintain service levels.
        
        4. **Optimization Opportunity**: Consider consolidating orders from the same suppliers to reduce shipping costs.
        
        5. **Cash Flow Tip**: Stagger non-urgent orders over {max(1, len(reorder_needed) // 10)} weeks to manage cash flow.
        """
        
        st.info(insights)
    
    else:
        st.success("✅ All inventory levels are optimal! No reorders needed at this time.")
        st.balloons()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>AI-Powered Inventory Optimizer v1.0 | Built with Streamlit & OpenAI</p>
        <p>Real-time analysis of your SCManagement inventory data</p>
    </div>
    """,
    unsafe_allow_html=True
)
