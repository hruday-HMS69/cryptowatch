use crate::error::CryptoWatchError;
use futures_util::{SinkExt, StreamExt};
pub use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite};

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";
const INITIAL_RECONNECT_DELAY: u64 = 1;
const MAX_RECONNECT_DELAY: u64 = 30;
const MAX_RETRIES: u32 = 10;

#[derive(Debug, Serialize, Deserialize)]
struct TickerData {
    s: String, // Symbol
    E: Option<i64>,
    c: String, // Last price
    p: String, // Price change
    P: String, // Price change percent
    v: String, // Volume
    q: String, // Quote volume
}

#[derive(Debug, Serialize, Clone, Deserialize)]
pub struct LatencyData {
    pub symbol: String,
    pub receive_time: i64,    // timestamp when message was received
    pub exchange_time: i64,   // timestamp from exchange if available
    pub processing_time: i64, // time taken to process the message
}

pub async fn connect_to_ticker(
    symbols: &[&str],
    sender: tokio::sync::mpsc::Sender<(String, String, String)>,
    latency_sender: tokio::sync::mpsc::Sender<LatencyData>,
) -> Result<(), CryptoWatchError> {
    let mut retry_count = 0;
    let mut reconnect_delay = INITIAL_RECONNECT_DELAY;

    info!("Starting WebSocket connection manager");

    loop {
        match try_connect(symbols, &sender, &latency_sender).await {
            Ok(_) => {
                info!("WebSocket connection closed gracefully");
                return Ok(());
            }
            Err(e) => {
                error!("WebSocket error: {}", e);

                if retry_count >= MAX_RETRIES {
                    error!("Maximum reconnection attempts ({}) reached", MAX_RETRIES);
                    return Err(e);
                }

                retry_count += 1;
                warn!(
                    "Attempting to reconnect in {} seconds (attempt {}/{})",
                    reconnect_delay, retry_count, MAX_RETRIES
                );

                sleep(Duration::from_secs(reconnect_delay)).await;

                // Exponential backoff with max limit
                reconnect_delay = (reconnect_delay * 2).min(MAX_RECONNECT_DELAY);
            }
        }
    }
}

async fn try_connect(
    symbols: &[&str],
    sender: &tokio::sync::mpsc::Sender<(String, String, String)>,
    latency_sender: &tokio::sync::mpsc::Sender<LatencyData>,
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

    ws_stream
        .send(tungstenite::Message::Text(subscribe_msg.to_string()))
        .await?;

    // Main processing loop
    while let Some(message) = ws_stream.next().await {
        let receive_time = chrono::Local::now().timestamp_millis();
        match message {
            Ok(tungstenite::Message::Text(text)) => {
                let processing_start = std::time::Instant::now();

                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if let Ok(ticker) = serde_json::from_str::<TickerData>(&text) {
                    let processing_time = processing_start.elapsed().as_millis() as i64;
                    let exchange_time = ticker.E.unwrap_or(receive_time);
                    log::debug!(
                    target: "latency",  // Special target for latency logs
                     "{} => {}ms (R:{} E:{})",
                     ticker.s,
                        processing_time,
                        receive_time,
                        exchange_time
                     );

                    let symbol = ticker.s.clone();
                    if sender.send((ticker.s, ticker.c, ticker.P)).await.is_err() {
                        break;
                    }
                    let processing_time = processing_start.elapsed().as_millis() as i64;
                    let _ = latency_sender
                        .send(LatencyData {
                            symbol,
                            receive_time,
                            exchange_time: receive_time,
                            processing_time,
                        })
                        .await;
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
