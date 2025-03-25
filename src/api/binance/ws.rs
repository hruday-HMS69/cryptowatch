use futures_util::StreamExt;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite;
use serde::{Deserialize, Serialize};
use log::{info, error, warn};
use crate::error::CryptoWatchError;

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443";

#[derive(Debug, Serialize, Deserialize)]
struct TickerData {
    stream: String,
    data: TickerDataInner,
}

#[derive(Debug, Serialize, Deserialize)]
struct TickerDataInner {
    s: String,  // Symbol
    c: String,  // Last price
    p: String,  // Price change
    P: String,  // Price change percent
    v: String,  // Volume
}

pub async fn connect_to_ticker(
    symbols: &[&str],
    sender: tokio::sync::mpsc::Sender<(String, String, String)>
) -> Result<(), CryptoWatchError> {
    // Create combined stream name
    let streams = symbols
        .iter()
        .map(|s| format!("{}@ticker", s.to_lowercase()))
        .collect::<Vec<_>>()
        .join("/");
    let url = format!("{}/stream?streams={}", BINANCE_WS_URL, streams);

    info!("Connecting to Binance WebSocket: {}", url);

    match connect_async(&url).await {
        Ok((mut ws_stream, _)) => {
            info!("Successfully connected to WebSocket");

            while let Some(message) = ws_stream.next().await {
                match message {
                    Ok(tungstenite::protocol::Message::Text(text)) => {
                        match serde_json::from_str::<TickerData>(&text) {
                            Ok(ticker) => {
                                info!(
                                    "{}: Price = {} (Change: {}%) Volume: {}",
                                    ticker.data.s, ticker.data.c, ticker.data.P, ticker.data.v
                                );

                                // Send data to dashboard
                                if let Err(e) = sender.send((
                                    ticker.data.s.clone(),
                                    ticker.data.c.clone(),
                                    format!("{}%", ticker.data.P)
                                )).await {
                                    error!("Failed to send to dashboard: {}", e);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse ticker data: {}", e);
                            }
                        }
                    }
                    Ok(tungstenite::protocol::Message::Close(_)) => {
                        info!("WebSocket connection closed");
                        break;
                    }
                    Ok(_) => {}  // Ignore other message types
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            error!("Connection error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}