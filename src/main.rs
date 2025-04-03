mod api;
mod error;
mod ui;

use std::io::Write;
use log::{info, LevelFilter};
use env_logger::Builder;
use api::binance::ws;
use std::error::Error;
use ui::dashboard::Dashboard;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Configure logger
    Builder::new()
        .filter_level(LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {}] {}",
                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .target(env_logger::Target::Stderr)
        .init();

    info!("Starting Cryptowatch Pro...");

    let symbols = [
        "btcusdt", "ethusdt", "solusdt", "xrpusdt",
        "adausdt", "dogeusdt", "dotusdt", "avaxusdt"
    ];

    let (tx, rx) = tokio::sync::mpsc::channel(100);
    let dashboard = Dashboard::new(10000.0);

    // Start WebSocket in separate task
    let ws_handle = tokio::spawn({
        let tx = tx.clone();
        async move {
            if let Err(e) = ws::connect_to_ticker(&symbols, tx).await {
                log::error!("WebSocket failed: {}", e);
            }
        }
    });

    // Run dashboard
    if let Err(e) = dashboard.run(rx).await {
        log::error!("Dashboard error: {}", e);
    }

    // Cleanup
    ws_handle.abort();
    let _ = ws_handle.await;

    info!("Shutdown complete");
    Ok(())
}
