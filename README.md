# Cryptowatch Terminal 📊💰

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A real-time cryptocurrency market monitor and trading dashboard built with Rust, featuring:

- 📈 Live price tracking (Binance API)
- 🚨 Custom price alerts
- 💼 Portfolio management
- 🔍 Technical indicators (RSI, MACD)
- ⌨️ Keyboard-driven TUI interface

![Dashboard Screenshot](screenshot.png) *(Example: Replace with actual screenshot)*

## Features

✔ **Real-time Data**  
   - Stream prices for BTC, ETH, SOL, XRP  
   - 24h change and volume tracking  

✔ **Portfolio Tracking**  
   - Value calculations  
   - Profit/Loss reporting  

✔ **Technical Analysis**  
   - RSI and MACD indicators  
   - Price history visualization  

✔ **Alerts System**  
   - Price threshold notifications  
   - Percentage change triggers  

## Installation

### Prerequisites
- Rust 1.70+
- Binance API key (optional for private endpoints)

```bash
# Clone repository
git clone https://github.com/yourusername/cryptowatch-terminal.git
cd cryptowatch-terminal

# Build and run
cargo run --release
