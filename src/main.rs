mod api;
mod error;
mod ui;

use log::{LevelFilter, error, info};
use env_logger::Builder;
use api::binance::ws;
use std::error::Error;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    Builder::new()
        .filter(None, LevelFilter::Info)
        .format_timestamp_secs()
        .format_module_path(false)
        .init();

    println!("🚀 Starting Cryptowatch Pro...");

    let symbols = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"];
    let (tx, rx) = tokio::sync::mpsc::channel(100);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let dashboard_handle = tokio::spawn(async move {
        match ui::dashboard::Dashboard::run(rx).await {
            Ok(_) => info!("Dashboard shutdown cleanly"),
            Err(e) => error!("Dashboard error: {}", e),
        }
    });

    let ws_handle = tokio::spawn({
        let tx = tx.clone();
        async move {
            match ws::connect_to_ticker(&symbols, tx).await {
                Ok(_) => info!("WebSocket shutdown cleanly"),
                Err(e) => error!("WebSocket error: {}", e),
            }
        }
    });

    let ctrl_c_handle = tokio::spawn(async move {
        signal::ctrl_c().await.ok();
        info!("Received shutdown signal");
        let _ = shutdown_tx.send(());
    });

    tokio::select! {
        _ = dashboard_handle => {},
        _ = ws_handle => {},
        _ = ctrl_c_handle => {},
        _ = shutdown_rx => {
            info!("Shutdown signal received");
        }
    }

    info!("Clean shutdown complete");
    Ok(())
}