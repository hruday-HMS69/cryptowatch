mod api;
mod error;
mod ui;

use api::binance::ws;
use env_logger::Builder;
use log::{info, LevelFilter};
use std::error::Error;
use std::io::Write;
use ui::dashboard::Dashboard;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Configure logger
    Builder::new()
        .filter_level(LevelFilter::Info) // Keep at Info for production
        .filter_module("cryptowatch", LevelFilter::Debug) // Your crate can debug
        .filter_module("latency", LevelFilter::Off) // Disable latency target logs
        .format(|buf, record| {
            // Skip debug logs from the TUI rendering
            if record.level() <= LevelFilter::Info {
                let ts = chrono::Local::now().format("%H:%M:%S%.3f");
                writeln!(
                    buf,
                    "[{} {:<5} {}] {}",
                    ts,
                    record.level(),
                    record.target(),
                    record.args()
                )
            } else {
                Ok(())
            }
        })
        .target(env_logger::Target::Stderr) // Keep logs separate from TUI
        .write_style(env_logger::WriteStyle::Always)
        .init();

    info!("Starting Cryptowatch Pro...");

    let symbols = [
        "btcusdt", "ethusdt", "solusdt", "xrpusdt", "adausdt", "dogeusdt", "dotusdt", "avaxusdt",
    ];

    // Create channels
    let (price_tx, price_rx) = tokio::sync::mpsc::channel(100);
    let (latency_tx, mut latency_rx) = tokio::sync::mpsc::channel(100);

    // Create dashboard
    let dashboard = Dashboard::new(10000.0);

    // Start WebSocket connection
    let ws_handle = tokio::spawn({
        let price_tx = price_tx.clone();
        async move {
            if let Err(e) = ws::connect_to_ticker(&symbols, price_tx, latency_tx).await {
                log::error!("WebSocket failed: {}", e);
            }
        }
    });

    // Start latency processing task
    let latency_handle = {
        let dashboard = dashboard.clone();
        tokio::spawn(async move {
            while let Some(latency) = latency_rx.recv().await {
                log::debug!("Latency Data Received: {:?}", latency);
                let symbol = latency.symbol.clone(); // Clone the symbol first
                let mut latency_data = dashboard.latency_data.lock().unwrap();

                // Get or create the VecDeque for this symbol
                let entries = latency_data.entry(symbol).or_default();

                // Push the latency data
                entries.push_back(latency);

                // Trim to keep only the last 100 entries
                if entries.len() > 100 {
                    entries.pop_front();
                }
            }
        })
    };

    // Run dashboard with price updates
    let dashboard_handle = tokio::spawn(async move {
        if let Err(e) = dashboard.run(price_rx).await {
            log::error!("Dashboard error: {}", e);
        }
    });

    // Wait for tasks to complete (they won't unless there's an error)
    tokio::select! {
        _ = ws_handle => {},
        _ = latency_handle => {},
        _ = dashboard_handle => {},
    };

    info!("Shutdown complete");
    Ok(())
}
