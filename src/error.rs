use thiserror::Error;

#[derive(Error, Debug)]
pub enum CryptoWatchError {
    #[error("WebSocket error: {0}")]
    WebsocketError(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}