# Live Implied Volatility 3D Surface

This repository contains the code of a **Streamlit** web application that dynamically fetches options market data, calculates **implied volatility** using the **Black-Scholes model**, and visualizes the **3D surface of implied volatility** and its corresponding Greeks (**Delta, Gamma, Vega, Theta, and Rho**).

## Features

- **Live data fetching** from Yahoo Finance via `yfinance`
- **Black-Scholes model implementation** for option pricing
- **Implied Volatility computation** via numerical root finding
- **Dynamic 3D plotting** of the implied volatility surface using **Plotly**
- **Greek calculations** (Delta, Gamma, Vega, Theta, Rho) for risk analysis
- **Interactive user inputs** for ticker selection, model parameters, and strike price range
- **Multiple tickers supported** (SPY, QQQ, NVDA, TSLA, AAPL, etc.)

---

## Access the Application

**Try it live**: [Implied Volatility & Greeks Surfaces](https://implied-volatility-dynamic-3d-surface.streamlit.app/)

No installation required! Simply visit the link and explore the **live implied volatility surface and Greeks visualization** directly in your browser.

---

## How It Works

1. The app fetches **real-time options chain data** from Yahoo Finance.
2. It computes the **implied volatility** for each strike price and expiration using the Black-Scholes model.
3. The **volatility surface is interpolated** and visualized as a **3D Plotly surface**.
4. The **Greeks (Delta, Gamma, Vega, Theta, Rho) are computed** and displayed in separate visualizations.

---

## Dependencies

The required dependencies are listed in `requirements.txt`.

---

## Usage Example

1. Select a **ticker** (SPY, AAPL, NVDA, etc.) from the sidebar.
2. Adjust **strike price range** and **Black-Scholes parameters**.
3. View **real-time implied volatility & Greeks** on interactive 3D plots.

---

## License

This project is open-source and available under the **MIT License**.

---

## Author

**Antonin Bezard**  
Reach me via [LinkedIn](https://www.linkedin.com/in/antonin-bezard-a11511177/)

---
