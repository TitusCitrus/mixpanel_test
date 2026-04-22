import streamlit as st
import pandas as pd
from backend import run_dynamic_forecast 
import plotly.graph_objects as go

# ==========================================
# 1. GAUGE CHART FUNCTION 
# ==========================================
def draw_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "<b>Market Sentiment</b>", 'font': {'size': 20}},
        gauge = {
            'axis': {
                'range': [0, 1], 
                'tickmode': 'array',
                'tickvals': [0.1, 0.3, 0.5, 0.7, 0.9],
                'ticktext': ["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"],
                'tickfont': {'size': 13, 'color': "black"}
            },
            'bar': {'color': "black", 'thickness': 0.2},
            'steps': [
                {'range': [0, 0.2], 'color': "#ff4d4d"},
                {'range': [0.2, 0.4], 'color': "#ffa64d"},
                {'range': [0.4, 0.6], 'color': "#e6e6e6"},
                {'range': [0.6, 0.8], 'color': "#99ff99"},
                {'range': [0.8, 1.0], 'color': "#33cc33"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    
    legend_colors = [
        ("Strong Bearish", "#ff4d4d"),
        ("Bearish", "#ffa64d"),
        ("Neutral", "#e6e6e6"),
        ("Bullish", "#99ff99"),
        ("Strong Bullish", "#33cc33")
    ]
    
    for label, color in legend_colors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], 
            mode='markers',
            marker=dict(color=color, symbol='square', size=15),
            name=label
        ))

    fig.update_layout(
        height=400, 
        margin=dict(t=60, b=50, l=40, r=40),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.1, 
            xanchor="center", 
            x=0.5
        )
    )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig

# ==========================================
# 2. INTERACTIVE PRICE CHART FUNCTION
# ==========================================
def draw_price_chart(df):
    df_recent = df.tail(90) 
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_recent.index,
        open=df_recent['Open'],
        high=df_recent['High'],
        low=df_recent['Low'],
        close=df_recent['Close'],
        name='Market Data'
    )])

    fig.update_layout(
        title={'text': "<b>Recent Price Action (Last 90 Days)</b>", 'font': {'size': 18}},
        yaxis_title="Price",
        xaxis_title="Date",
        hovermode="x unified",
        xaxis_rangeslider_visible=False, 
        height=400,
        margin=dict(t=50, b=20, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.2)')
    return fig

# ==========================================
# 3. BROKER INVENTORY CHART FUNCTION 
# ==========================================
def draw_broker_inventory_chart(broker_file):
    df = pd.read_csv(broker_file)
    
    def parse_val(val):
        if pd.isna(val): return 0.0
        val_str = str(val).replace('"', '').replace(',', '').strip()
        if 'T' in val_str: return float(val_str.replace('T', '')) * 1e12
        elif 'B' in val_str: return float(val_str.replace('B', '')) * 1e9
        elif 'M' in val_str: return float(val_str.replace('M', '')) * 1e6
        elif 'K' in val_str: return float(val_str.replace('K', '')) * 1e3
        else:
            try: return float(val_str)
            except: return 0.0

    if not all(col in df.columns for col in ['BY', 'B.val', 'SL', 'S.val']):
        return None 

    df['B.val'] = df['B.val'].apply(parse_val)
    df['S.val'] = df['S.val'].apply(parse_val)
    
    buys = df.groupby('BY')['B.val'].sum().reset_index().rename(columns={'BY': 'Broker', 'B.val': 'Buy'})
    sells = df.groupby('SL')['S.val'].sum().reset_index().rename(columns={'SL': 'Broker', 'S.val': 'Sell'})
    
    unified = pd.merge(buys, sells, on='Broker', how='outer').fillna(0)
    unified['Total_Vol'] = unified['Buy'] + unified['Sell']
    unified['Net_Vol'] = unified['Buy'] - unified['Sell']
    
    top_10 = unified.nlargest(10, 'Total_Vol').sort_values('Net_Vol')
    
    colors = ['#ff4d4d' if val < 0 else '#33cc33' for val in top_10['Net_Vol']]
    
    fig = go.Figure(go.Bar(
        x=top_10['Net_Vol'],
        y=top_10['Broker'],
        orientation='h',
        marker_color=colors,
        text=top_10['Net_Vol'].apply(lambda x: f"Rp {x/1e9:.1f}B"), 
        textposition='auto'
    ))
    
    fig.update_layout(
        title={'text': "<b>Top 10 Brokers: Net Inventory</b>", 'font': {'size': 18}},
        xaxis_title="Net Volume (Rp)",
        yaxis_title="Broker Code",
        height=400,
        margin=dict(t=50, b=20, l=40, r=40),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.2)')
    return fig

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Trading Forecast", page_icon="📈", layout="wide")

st.title("Dynamic Algorithmic Price Prediction Model")

st.sidebar.header("Configuration")

st.sidebar.subheader("1. Upload Data")
target_file = st.sidebar.file_uploader("Upload Target Asset", type=['csv'])
broker_file = st.sidebar.file_uploader("Upload Broker Summary", type=['csv']) 

st.sidebar.subheader("2. Prediction Parameters")
lookahead_input = st.sidebar.slider("Forecast Horizon (Days Ahead)", min_value=1, max_value=14, value=3)
atr_input = st.sidebar.slider("ATR Safety Multiplier (Volatility Buffer)", min_value=0.1, max_value=3.0, value=1.5, step=0.1)

if target_file and broker_file:
    st.info(f"Price and Broker data loaded successfully. Ready to generate a {lookahead_input}-day forecast.")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        target_file.seek(0)
        chart_df = pd.read_csv(target_file)
        if 'Date' in chart_df.columns:
            chart_df['Date'] = pd.to_datetime(chart_df['Date'])
            chart_df.set_index('Date', inplace=True)
        st.plotly_chart(draw_price_chart(chart_df), use_container_width=True)
        
    with col_chart2:
        broker_file.seek(0)
        inventory_chart = draw_broker_inventory_chart(broker_file)
        if inventory_chart:
            st.plotly_chart(inventory_chart, use_container_width=True)
        else:
            st.warning("Could not parse broker columns. Ensure 'BY', 'B.val', 'SL', 'S.val' exist.")

    target_file.seek(0) 
    broker_file.seek(0)
    
    if st.button("🚀 Run Forecast"):
        with st.spinner(f"Fusing order flow data, training models, and predicting {lookahead_input} days ahead..."):
            try:
                results_df, latest_score = run_dynamic_forecast(
                    target_csv=target_file, 
                    broker_csv=broker_file,
                    lookahead_days=lookahead_input, 
                    atr_multiplier=atr_input
                )
                
                st.success("Forecast Generated Successfully!")
                st.divider() 
                
                latest_pred = results_df.iloc[-1]
                
                col1, col2 = st.columns([1, 1.5])
                
                with col1:
                    st.plotly_chart(draw_gauge(latest_score), use_container_width=True)
                    
                with col2:
                    st.markdown(f"### 📅 Forecast Target Date: **{latest_pred['Forecast_Date']}**")
                    
                    score = latest_pred['Sentiment_Score']
                    if score >= 0.8: readable_sentiment = f"{score} Strong Bullish"
                    elif score >= 0.6: readable_sentiment = f"{score} Bullish"
                    elif score > 0.4: readable_sentiment = f"{score} Neutral"
                    elif score > 0.2: readable_sentiment = f"{score} Bearish"
                    else: readable_sentiment = f"{score} Strong Bearish"

                    st.markdown(f"#### 🧠 AI Sentiment: **{readable_sentiment}**")
                    st.write(f"*This AI has dynamically trained itself on the past price action and broker order flow to predict {lookahead_input} trading days into the future. The price zones below include a {atr_input}x ATR buffer to account for current market volatility.*")
                
                st.write("") 
                st.markdown("### 📊 Actionable Trading Plan")
                
                card1, card2, card3 = st.columns(3)
                
                with card1:
                    st.info(f"📥 **TARGET ENTRY**\n\n**{latest_pred['Target_Buy'].split('(')[0].strip()}**\n\n*(Safe Zone: {latest_pred['Target_Buy'].split('(')[1]}*")
                with card2:
                    st.success(f"💰 **TAKE PROFIT**\n\n**{latest_pred['Take_Profit'].split('(')[0].strip()}**\n\n*(Exit Zone: {latest_pred['Take_Profit'].split('(')[1]}*")
                with card3:
                    st.error(f"🛡️ **STOP LOSS**\n\n**{latest_pred['Stop_Loss'].split('(')[0].strip()}**\n\n*(Danger Zone: {latest_pred['Stop_Loss'].split('(')[1]}*")
                
                st.divider()
                
            except Exception as e:
                st.error(f"An error occurred in the backend: {e}")
elif target_file or broker_file:
    st.warning("👈 Please upload BOTH the Target Asset and Broker Summary CSVs to begin.")
else:
    st.info("👈 Please upload your CSV datasets in the sidebar to begin.")