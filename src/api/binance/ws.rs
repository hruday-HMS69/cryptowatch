use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite};
use serde::{Deserialize, Serialize};
use log::{error, warn};
use crate::error::CryptoWatchError;

const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws";

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
    let (mut ws_stream, _) = connect_async(BINANCE_WS_URL).await?;

    for symbol in symbols {
        let subscribe_msg = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": [format!("{}@ticker", symbol.to_lowercase())],
            "id": 1
        });
        ws_stream.send(tungstenite::Message::Text(subscribe_msg.to_string())).await?;
    }

    while let Some(message) = ws_stream.next().await {
        match message {
            Ok(tungstenite::Message::Text(text)) => {
                if let Ok(ticker) = serde_json::from_str::<TickerData>(&text) {
                    if let Err(e) = sender.send((
                        ticker.s,
                        ticker.c,
                        ticker.P
                    )).await {
                        error!("Failed to send to dashboard: {}", e);
                        break;
                    }
                }
            }
            Ok(tungstenite::Message::Close(_)) => {
                warn!("WebSocket connection closed");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
