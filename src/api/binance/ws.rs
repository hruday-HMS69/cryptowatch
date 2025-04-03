use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};
use crate::error::CryptoWatchError;
pub use log::{error, warn, info};

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";
const INITIAL_RECONNECT_DELAY: u64 = 1;
const MAX_RECONNECT_DELAY: u64 = 30;
const MAX_RETRIES: u32 = 10;

#[derive(Debug, Serialize, Deserialize)]
struct TickerData {
    s: String,  // Symbol
    c: String,  // Last price
    p: String,  // Price change
    P: String,  // Price change percent
    v: String,  // Volume
    q: String,  // Quote volume
}

pub async fn connect_to_ticker(
    symbols: &[&str],
    sender: tokio::sync::mpsc::Sender<(String, String, String)>
) -> Result<(), CryptoWatchError> {
    let mut retry_count = 0;
    let mut reconnect_delay = INITIAL_RECONNECT_DELAY;

    info!("Starting WebSocket connection manager");

    loop {
        match try_connect(symbols, &sender).await {
            Ok(_) => {
                info!("WebSocket connection closed gracefully");
                return Ok(());
            },
            Err(e) => {
                error!("WebSocket error: {}", e);

                if retry_count >= MAX_RETRIES {
                    error!("Maximum reconnection attempts ({}) reached", MAX_RETRIES);
                    return Err(e);
                }

                retry_count += 1;
                warn!("Attempting to reconnect in {} seconds (attempt {}/{})",
                    reconnect_delay, retry_count, MAX_RETRIES);

                sleep(Duration::from_secs(reconnect_delay)).await;

                // Exponential backoff with max limit
                reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
            }
        }
    }
}

async fn try_connect(
    symbols: &[&str],
    sender: &tokio::sync::mpsc::Sender<(String, String, String)>
) -> Result<(), CryptoWatchError> {
    info!("Connecting to Binance WebSocket at {}", BINANCE_WS_URL);
    let (mut ws_stream, _) = connect_async(BINANCE_WS_URL).await?;
    info!("Successfully connected to Binance WebSocket");

    // Subscribe to symbols
    let subscribe_msg = serde_json::json!({
        "method": "SUBSCRIBE",
        "params": symbols.iter()
            .map(|s| format!("{}@ticker", s.to_lowercase()))
            .collect::<Vec<_>>(),
        "id": 1
    });

    ws_stream.send(tungstenite::Message::Text(subscribe_msg.to_string())).await?;

    // Main processing loop
    while let Some(message) = ws_stream.next().await {
        match message {
            Ok(tungstenite::Message::Text(text)) => {
                if let Ok(ticker) = serde_json::from_str::<TickerData>(&text) {
                    if sender.send((ticker.s, ticker.c, ticker.P)).await.is_err() {
                        break;
                    }
                }
            }
            Ok(tungstenite::Message::Close(_)) => {
                break;
            }
            Ok(tungstenite::Message::Ping(ping)) => {
                let _ = ws_stream.send(tungstenite::Message::Pong(ping)).await;
            }
            Err(e) => {
                return Err(e.into());
            }
            _ => {}
        }
    }
    
    let _ = ws_stream.close(None).await;
    Ok(())
}
