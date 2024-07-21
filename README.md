# Real-Time-Market-Insights

## FILE NAVIGATION:

- `Streamlit_app`: Main folder which contains `datasets` and streamlit `app_stockonauts.py` .
    - `datasets`: Contains all datasets used in the project including `TickersData.csv` and `nifty data` .
    - `static`: All static Images used in the website
    - `app_stockonauts.py`: Streamlit app
 
- `Algorithms_ipynb`:Folder containing ipynb files for 3 algorithms introduced in this project.
    - `1.Alg_trading.ipynb`: Code for the algorithm which used NIM, NPA, P/E Ratio .
    - `2.Mutli_fac_market_fin_bert.ipynb`: ALgorithm which uses High Performance Colaborations,Digitilization Impacts, MIcro & Macro Economics and NLP Sentiment analysis for news sector wise.
    - `3.Personal_invest_ideas_algo.ipynb`: Code for Personalized stock investment strategy by  collecting user preferences for investment range, duration and sectors of interest .



## Streamlit UI Sample Images:

![image](https://github.com/Karthick-ng/Real-Time-Market-Insights/assets/116434132/1f6d8395-96f0-4d7f-8efe-42e9336ede0d)
![image](https://github.com/Karthick-ng/Real-Time-Market-Insights/assets/116434132/bd0d18ba-2843-4e8c-af7c-654621f6d403)

## UI Demo Video:
[![Here is the DEMO Video](https://github.com/Karthick-ng/Real-Time-Market-Insights/assets/116434132/1f6d8395-96f0-4d7f-8efe-42e9336ede0d)](https://www.youtube.com/watch?v=Q05lviXkE5M)

## Development phase:

## UI

### React-FrontEnd
![WhatsApp Image 2024-01-27 at 09 55 21_103a2329](https://github.com/Karthick-ng/Real-Time-Market-Insights/assets/116434132/1eb3300c-6880-478e-bfc8-a50b5ab4a4dc)

### Flask Backend
##### POSTMAN API -- POST & GET Request
![WhatsApp Image 2024-01-27 at 13 04 39_4a02391c](https://github.com/Karthick-ng/Real-Time-Market-Insights/assets/116434132/d3663934-a398-4623-8ecb-5afaa33183f4)

## Project Overview

The Real-Time Market Insights Platform is a comprehensive solution that combines algorithmic trading, multifactor market dynamics analysis, sentiment analysis using Natural Language Processing (NLP), and personalized investment insights. This project leverages advanced data science techniques to provide users with a holistic view of market trends and assist in making informed investment decisions.

## Project Components

### 1. Algorithmic Trading Strategy (Strategy 1)

#### Focus Area: Banking Sector Analysis

**Metrics Considered:**
- Higher Net Interest Margin (NIM)
- Lower Non-Performing Assets (NPA)
- Pricing and Earnings Analysis (P/E Ratio and P/B Ratio)

### 2. Multifactor Market Dynamics (Strategy 2)

#### Collaborative Approach and Digitalization Analysis:

- Evaluate collaborations in diverse sectors (e.g., tech-automotive, pharma-tech).
- Analyze the impact of digitalization on banking stocks, including fintech collaborations and blockchain integration.

#### Micro and Macro Economics:

- Explore microeconomic factors (individual markets) and macroeconomic factors (national-level markets).
- Assess the influence of government policies and global economic interdependencies.

### 3. NLP - Sentiment Analysis (Strategy 2)

**Motivation:**
Combine Fundamental and Technical Analysis with sentiment analysis to enhance stock selection.

**Data Collection:**
- Scraping articles from sources like the Economic Times.
- Extracting features such as article date, headline, summary, and URL.

**NLP Techniques:**
- Preprocessing techniques (lemmatization, etc.).
- Sentiment analysis to classify news articles as positive or negative.

### 4. Quarterly Earnings Analysis (Strategy 2)

**Process:**
- Monitor and analyze quarterly earnings reports for insights.
- Adjust strategy based on emerging trends in individual companies.
- Backtest the strategy using historical data.
- Assess performance under various market conditions.

### 5. Personalized Investment Insights (Strategy 3)

**User-Centric Approach:**
- Collect user preferences for investment range, duration, and sectors of interest.
- Validate user inputs and retrieve historical stock data for a predefined list of banks.
- Utilize the Prophet time-series forecasting model to predict future stock prices.
- Rank stocks based on predicted profits and present results to the user.
- Integrate with a language model to generate personalized investment recommendations.

## Technologies Used

- **Data Visualization:**
  - `matplotlib`
  - `plotly`

- **Web App Development:**
  - `streamlit`
  - `streamlit_option_menu`

- **Data Manipulation:**
  - `pandas`
  - `numpy`

- **Time Series Forecasting:**
  - `prophet`

- **Financial Data Retrieval:**
  - `yfinance`

- **Web Scraping and API Requests:**
  - `requests`
  - `beautifulsoup4`

- **Natural Language Processing (NLP):**
  - `transformers`
  - `bs4`

- **Miscellaneous:**
  - `replicate`
  - `random`
  - `cufflinks`

## Outcome

The Real-Time Market Insights Platform aims to provide users with a holistic view of market dynamics, sentiment analysis, and personalized investment insights. By combining multiple strategies and leveraging advanced Ai and data science techniques, the platform assists investors in making informed decisions in the ever-changing landscape of the stock market. The project emphasizes user engagement, real-time data analysis, and the integration of cutting-edge technologies to enhance the overall investment experience.

## Getting Started

1. Clone the repository.
   ```bash
   git clone [repository_url]
   
2. Move into project directory.
    ```bash
    cd Real-Time-Market-Insights
    cd Streamlit_app

3. Create a Virtual Environment and install Requirements.
    ```bash
    python -m venv venv
    venv\Scripts\activate  (on windows)
    source venv/bin/activate (on MAC)
  
    pip install -r requirements.txt

4. Run Streamlit code.
   ```bash
   streamlit run app_stockonauts.py

