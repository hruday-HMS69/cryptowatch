use crate::api::binance::ws::LatencyData;
use chrono::Local;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::Alignment;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Paragraph, Row, Sparkline, Table},
    Frame, Terminal,
};
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::io;
use std::sync::{Arc, Mutex};
use ta::{
    indicators::{
        BollingerBands, ExponentialMovingAverage, MovingAverageConvergenceDivergence,
        RelativeStrengthIndex, SimpleMovingAverage,
    },
    Next,
};

type DynError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardView {
    Prices,
    Alerts,
    Portfolio,
    Exchanges,
    Indicators,
    Chart,
    Candles,
    Latency,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatencyDisplayMode {
    LineChart,
    Sparkline,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    PriceAbove(f64),
    PriceBelow(f64),
    ChangeAbove(f64),
    ChangeBelow(f64),
    IndicatorCross(String, f64),
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub symbol: String,
    pub condition: AlertCondition,
    pub triggered: bool,
    pub active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exchange {
    Binance,
    Coinbase,
    Kraken,
}

#[derive(Debug, Clone)]
pub struct CandleData {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct PriceData {
    pub price: f64,
    pub volume: f64,
    pub timestamp: i64,
    pub exchange: Exchange,
}

#[derive(Debug, Clone)]
pub struct IndicatorData {
    pub sma_20: Option<f64>,
    pub ema_20: Option<f64>,
    pub rsi_14: Option<f64>,
    pub macd: Option<f64>,
    pub macd_signal: Option<f64>,
    pub bollinger_upper: Option<f64>,
    pub bollinger_lower: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SymbolData {
    pub prices: VecDeque<PriceData>,
    pub candles: VecDeque<CandleData>,
    pub indicators: IndicatorData,
}

impl SymbolData {
    pub fn new() -> Self {
        Self {
            prices: VecDeque::with_capacity(100),
            candles: VecDeque::with_capacity(50),
            indicators: IndicatorData {
                sma_20: None,
                ema_20: None,
                rsi_14: None,
                macd: None,
                macd_signal: None,
                bollinger_upper: None,
                bollinger_lower: None,
            },
        }
    }

    pub fn update_indicators(&mut self) {
        if self.prices.len() < 20 {
            return;
        }

        let prices: Vec<f64> = self.prices.iter().map(|p| p.price).collect();

        // SMA 20
        let mut sma = SimpleMovingAverage::new(20).unwrap();
        self.indicators.sma_20 = prices.iter().fold(None, |_, &p| Some(sma.next(p)));

        // EMA 20
        let mut ema = ExponentialMovingAverage::new(20).unwrap();
        self.indicators.ema_20 = prices.iter().fold(None, |_, &p| Some(ema.next(p)));

        // RSI 14
        let mut rsi = RelativeStrengthIndex::new(14).unwrap();
        self.indicators.rsi_14 = prices.iter().fold(None, |_, &p| Some(rsi.next(p)));

        // MACD
        let mut macd = MovingAverageConvergenceDivergence::new(12, 26, 9).unwrap();
        let macd_output = prices.iter().fold(None, |_, &p| Some(macd.next(p)));
        if let Some(macd_output) = macd_output {
            self.indicators.macd = Some(macd_output.macd);
            self.indicators.macd_signal = Some(macd_output.signal);
        }

        // Bollinger Bands
        let mut bb = BollingerBands::new(20, 2.0).unwrap();
        let bb_output = prices.iter().fold(None, |_, &p| Some(bb.next(p)));
        if let Some(bb_output) = bb_output {
            self.indicators.bollinger_upper = Some(bb_output.upper);
            self.indicators.bollinger_lower = Some(bb_output.lower);
        }
    }

    pub fn update_candles(&mut self, timeframe: Timeframe) {
        if self.prices.is_empty() {
            return;
        }

        let now = Local::now().timestamp();
        let timeframe_secs = timeframe.as_secs();

        let mut current_candle_start = (now / timeframe_secs) * timeframe_secs;
        let mut current_candle = CandleData {
            open: self.prices[0].price,
            high: self.prices[0].price,
            low: self.prices[0].price,
            close: self.prices[0].price,
            volume: self.prices[0].volume,
            timestamp: current_candle_start,
        };

        for price in &self.prices {
            let price_time = price.timestamp;
            let candle_start = (price_time / timeframe_secs) * timeframe_secs;

            if candle_start != current_candle_start {
                self.candles.push_back(current_candle);
                if self.candles.len() > 50 {
                    self.candles.pop_front();
                }

                current_candle_start = candle_start;
                current_candle = CandleData {
                    open: price.price,
                    high: price.price,
                    low: price.price,
                    close: price.price,
                    volume: price.volume,
                    timestamp: current_candle_start,
                };
            } else {
                current_candle.high = current_candle.high.max(price.price);
                current_candle.low = current_candle.low.min(price.price);
                current_candle.close = price.price;
                current_candle.volume += price.volume;
            }
        }

        self.candles.push_back(current_candle);
        if self.candles.len() > 50 {
            self.candles.pop_front();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeframe {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
    W1,
}

impl Timeframe {
    pub fn as_secs(&self) -> i64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1h",
            Timeframe::H4 => "4h",
            Timeframe::D1 => "1d",
            Timeframe::W1 => "1w",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub holdings: HashMap<String, f64>,
    pub initial_value: f64,
    pub cash: f64,
    pub trade_history: Vec<Trade>,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,
    pub trade_type: TradeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeType {
    Buy,
    Sell,
}

impl Portfolio {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            holdings: HashMap::new(),
            initial_value: initial_cash,
            cash: initial_cash,
            trade_history: Vec::new(),
        }
    }

    pub fn current_value(&self, prices: &HashMap<String, SymbolData>) -> f64 {
        self.holdings.iter().fold(0.0, |acc, (symbol, qty)| {
            prices
                .get(symbol)
                .and_then(|data| data.prices.back())
                .map(|price_data| price_data.price * qty)
                .unwrap_or(0.0)
                + acc
        }) + self.cash
    }

    pub fn profit_loss(&self, prices: &HashMap<String, SymbolData>) -> f64 {
        self.current_value(prices) - self.initial_value
    }

    pub fn buy(&mut self, symbol: &str, amount: f64, price: f64) -> Result<(), String> {
        let cost = amount * price;
        if cost > self.cash {
            return Err("Insufficient funds".to_string());
        }

        self.cash -= cost;
        *self.holdings.entry(symbol.to_string()).or_insert(0.0) += amount;

        self.trade_history.push(Trade {
            symbol: symbol.to_string(),
            quantity: amount,
            price,
            timestamp: Local::now().timestamp(),
            trade_type: TradeType::Buy,
        });

        Ok(())
    }

    pub fn sell(&mut self, symbol: &str, amount: f64, price: f64) -> Result<(), String> {
        let current_amount = *self.holdings.get(symbol).unwrap_or(&0.0);
        if amount > current_amount {
            return Err("Insufficient holdings".to_string());
        }

        self.cash += amount * price;
        *self.holdings.get_mut(symbol).unwrap() -= amount;

        if self.holdings[symbol] <= 0.0001 {
            self.holdings.remove(symbol);
        }

        self.trade_history.push(Trade {
            symbol: symbol.to_string(),
            quantity: amount,
            price,
            timestamp: Local::now().timestamp(),
            trade_type: TradeType::Sell,
        });

        Ok(())
    }
}

#[derive(Clone)]
pub struct Dashboard {
    pub prices: Arc<Mutex<HashMap<String, SymbolData>>>,
    pub selected: Arc<Mutex<usize>>,
    pub sort_column: Arc<Mutex<usize>>,
    pub sort_ascending: Arc<Mutex<bool>>,
    pub alerts: Arc<Mutex<Vec<Alert>>>,
    pub portfolio: Arc<Mutex<Portfolio>>,
    pub current_view: Arc<Mutex<DashboardView>>,
    pub price_history_length: usize,
    pub running: Arc<Mutex<bool>>,
    pub selected_timeframe: Arc<Mutex<Timeframe>>,
    pub ws_connected: Arc<Mutex<bool>>,
    pub latency_data: Arc<Mutex<HashMap<String, VecDeque<LatencyData>>>>,
    pub latency_display_mode: Arc<Mutex<LatencyDisplayMode>>,
    pub show_latency_thresholds: Arc<Mutex<bool>>,
    pub latency_zoom: Arc<Mutex<u8>>, // 0=normal, 1=zoomed, 2=max zoom
}

impl Dashboard {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            prices: Arc::new(Mutex::new(HashMap::new())),
            selected: Arc::new(Mutex::new(0)),
            sort_column: Arc::new(Mutex::new(0)),
            sort_ascending: Arc::new(Mutex::new(false)),
            alerts: Arc::new(Mutex::new(vec![
                Alert {
                    symbol: "BTCUSDT".to_string(),
                    condition: AlertCondition::PriceAbove(90000.0),
                    triggered: false,
                    active: true,
                },
                Alert {
                    symbol: "ETHUSDT".to_string(),
                    condition: AlertCondition::PriceBelow(2000.0),
                    triggered: false,
                    active: true,
                },
            ])),
            portfolio: Arc::new(Mutex::new(Portfolio::new(initial_cash))),
            current_view: Arc::new(Mutex::new(DashboardView::Prices)),
            price_history_length: 100,
            running: Arc::new(Mutex::new(true)),
            selected_timeframe: Arc::new(Mutex::new(Timeframe::M1)),
            ws_connected: Arc::new(Mutex::new(false)),
            latency_data: Arc::new(Mutex::new(HashMap::new())),
            latency_display_mode: Arc::new(Mutex::new(LatencyDisplayMode::LineChart)),
            show_latency_thresholds: Arc::new(Mutex::new(true)),
            latency_zoom: Arc::new(Mutex::new(0)),
        }
    }

    pub fn is_connected(&self) -> bool {
        *self.ws_connected.lock().unwrap()
    }

    pub async fn run(
        &self,
        mut receiver: tokio::sync::mpsc::Receiver<(String, String, String)>,
    ) -> Result<(), DynError> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        *self.ws_connected.lock().unwrap() = true;

        while *self.running.lock().unwrap() {
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key_input(key).await?;
                }
            }

            while let Ok((symbol, price, change)) = receiver.try_recv() {
                if let Err(e) = self.update_price(&symbol, &price, &change).await {
                    log::error!("Error updating price for {}: {}", symbol, e);
                }
                self.check_alerts(&symbol).await?;
            }

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(5),
                        Constraint::Length(3),
                    ])
                    .split(f.size());

                self.render_header(f, chunks[0]);
                self.render_main_content(f, chunks[1]);
                self.render_footer(f, chunks[2]);
            })?;
        }

        *self.ws_connected.lock().unwrap() = false;
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    async fn handle_key_input(&self, key: KeyEvent) -> Result<(), DynError> {
        match key.code {
            KeyCode::Char('q') => *self.running.lock().unwrap() = false,
            KeyCode::Char('p') => *self.current_view.lock().unwrap() = DashboardView::Portfolio,
            KeyCode::Char('a') => *self.current_view.lock().unwrap() = DashboardView::Alerts,
            KeyCode::Char('e') => *self.current_view.lock().unwrap() = DashboardView::Exchanges,
            KeyCode::Char('i') => *self.current_view.lock().unwrap() = DashboardView::Indicators,
            KeyCode::Char('c') => *self.current_view.lock().unwrap() = DashboardView::Chart,
            KeyCode::Char('k') => *self.current_view.lock().unwrap() = DashboardView::Candles,
            KeyCode::Char('v') => *self.current_view.lock().unwrap() = DashboardView::Prices,
            KeyCode::Char('l') => *self.current_view.lock().unwrap() = DashboardView::Latency,
            KeyCode::Left => {
                let mut selected = self.selected.lock().unwrap();
                *selected = selected.saturating_sub(1);
            }
            KeyCode::Right => {
                let mut selected = self.selected.lock().unwrap();
                *selected = selected.saturating_add(1);
            }
            KeyCode::Char('t') => {
                let current_view = *self.current_view.lock().unwrap();
                if current_view == DashboardView::Latency {
                    // Toggle display mode in latency view
                    let mut display_mode = self.latency_display_mode.lock().unwrap();
                    *display_mode = match *display_mode {
                        LatencyDisplayMode::LineChart => LatencyDisplayMode::Sparkline,
                        LatencyDisplayMode::Sparkline => LatencyDisplayMode::LineChart,
                    };
                } else {
                    // Switch timeframe in other views
                    let mut timeframe = self.selected_timeframe.lock().unwrap();
                    *timeframe = match *timeframe {
                        Timeframe::M1 => Timeframe::M5,
                        Timeframe::M5 => Timeframe::M15,
                        Timeframe::M15 => Timeframe::M30,
                        Timeframe::M30 => Timeframe::H1,
                        Timeframe::H1 => Timeframe::H4,
                        Timeframe::H4 => Timeframe::D1,
                        Timeframe::D1 => Timeframe::W1,
                        Timeframe::W1 => Timeframe::M1,
                    };
                }
        }
            KeyCode::Char('h') => {
                // Show/hide threshold lines
                let mut show_thresholds = self.show_latency_thresholds.lock().unwrap();
                *show_thresholds = !*show_thresholds;
            }
            KeyCode::Char('z') => {
                // Zoom in/out
                let mut zoom = self.latency_zoom.lock().unwrap();
                *zoom = (*zoom + 1) % 3; // Cycles through 0,1,2 zoom levels
            }
            KeyCode::Up => {
                let mut selected = self.selected.lock().unwrap();
                *selected = selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let mut selected = self.selected.lock().unwrap();
                *selected = selected.saturating_add(1);
            }
            KeyCode::Char('s') => {
                let mut sort_column = self.sort_column.lock().unwrap();
                *sort_column = (*sort_column + 1) % 3;
                let mut sort_ascending = self.sort_ascending.lock().unwrap();
                *sort_ascending = !*sort_ascending;
            }


            KeyCode::Char('b') => {
                let prices = self.prices.lock().unwrap();
                let selected = *self.selected.lock().unwrap();
                if let Some(symbol) = prices.keys().nth(selected) {
                    let mut portfolio = self.portfolio.lock().unwrap();
                    if let Some(data) = prices.get(symbol) {
                        if let Some(price_data) = data.prices.back() {
                            let amount = 0.1;
                            let price = price_data.price;
                            if let Err(e) = portfolio.buy(symbol, amount, price) {
                                log::error!("Buy error: {}", e);
                            }
                        }
                    }
                }
            }
            KeyCode::Char('x') => {
                let prices = self.prices.lock().unwrap();
                let selected = *self.selected.lock().unwrap();
                if let Some(symbol) = prices.keys().nth(selected) {
                    let mut portfolio = self.portfolio.lock().unwrap();
                    if let Some(data) = prices.get(symbol) {
                        if let Some(price_data) = data.prices.back() {
                            let amount = 0.1;
                            let price = price_data.price;
                            if let Err(e) = portfolio.sell(symbol, amount, price) {
                                log::error!("Sell error: {}", e);
                            }
                        }
                    }
                }
            }
            _ => (),
        }
        Ok(())
    }

    async fn update_price(&self, symbol: &str, price: &str, change: &str) -> Result<(), DynError> {
        let price = match price.parse::<f64>() {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to parse price '{}' for {}: {}", price, symbol, e);
                return Ok(());
            }
        };

        let _change = match change.parse::<f64>() {
            Ok(c) => c,
            Err(e) => {
                log::error!("Failed to parse change '{}' for {}: {}", change, symbol, e);
                return Ok(());
            }
        };

        let timestamp = Local::now().timestamp();

        let price_data = PriceData {
            price,
            volume: 0.0,
            timestamp,
            exchange: Exchange::Binance,
        };

        let mut prices = self.prices.lock().unwrap();
        let symbol_data = prices
            .entry(symbol.to_uppercase())
            .or_insert_with(SymbolData::new);

        symbol_data.prices.push_back(price_data);
        if symbol_data.prices.len() > self.price_history_length {
            symbol_data.prices.pop_front();
        }

        symbol_data.update_indicators();
        symbol_data.update_candles(*self.selected_timeframe.lock().unwrap());

        Ok(())
    }

    async fn check_alerts(&self, symbol: &str) -> Result<(), DynError> {
        let prices = self.prices.lock().unwrap();
        let symbol_data = match prices.get(&symbol.to_uppercase()) {
            Some(data) => data,
            None => return Ok(()),
        };

        let price_data = match symbol_data.prices.back() {
            Some(data) => data,
            None => return Ok(()),
        };

        let mut alerts = self.alerts.lock().unwrap();
        for alert in alerts.iter_mut() {
            if alert.symbol != symbol.to_uppercase() || !alert.active {
                continue;
            }

            let should_trigger = match &alert.condition {
                AlertCondition::PriceAbove(threshold) => price_data.price > *threshold,
                AlertCondition::PriceBelow(threshold) => price_data.price < *threshold,
                AlertCondition::ChangeAbove(threshold) => {
                    if symbol_data.prices.len() >= 2 {
                        let prev_price = symbol_data.prices[symbol_data.prices.len() - 2].price;
                        let change = (price_data.price - prev_price) / prev_price * 100.0;
                        change > *threshold
                    } else {
                        false
                    }
                }
                AlertCondition::ChangeBelow(threshold) => {
                    if symbol_data.prices.len() >= 2 {
                        let prev_price = symbol_data.prices[symbol_data.prices.len() - 2].price;
                        let change = (price_data.price - prev_price) / prev_price * 100.0;
                        change < *threshold
                    } else {
                        false
                    }
                }
                AlertCondition::IndicatorCross(indicator, threshold) => match indicator.as_str() {
                    "RSI" => symbol_data
                        .indicators
                        .rsi_14
                        .map_or(false, |rsi| rsi > *threshold),
                    "MACD" => symbol_data
                        .indicators
                        .macd
                        .zip(symbol_data.indicators.macd_signal)
                        .map_or(false, |(macd, signal)| macd > signal),
                    "BB" => {
                        price_data.price > symbol_data.indicators.bollinger_upper.unwrap_or(0.0)
                            || price_data.price
                            < symbol_data.indicators.bollinger_lower.unwrap_or(0.0)
                    }
                    _ => false,
                },
            };
            alert.triggered = should_trigger;
        }
        Ok(())
    }

    fn render_header(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let portfolio = self.portfolio.lock().unwrap();
        let prices = self.prices.lock().unwrap();

        let connection_status = if self.is_connected() {
            Span::styled("CONNECTED", Style::default().fg(Color::Green))
        } else {
            Span::styled("DISCONNECTED", Style::default().fg(Color::Red))
        };

        let header = Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::styled(
                    "CRYPTOWATCH ",
                    Style::default()
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                ),
                connection_status,
            ]),
            Line::from(Span::styled(
                format!(
                    "Last update: {} | Portfolio: ${:.2} | Timeframe: {}",
                    Local::now().format("%H:%M:%S"),
                    portfolio.current_value(&prices),
                    self.selected_timeframe.lock().unwrap().as_str()
                ),
                Style::default().fg(Color::Gray),
            )),
        ]))
            .block(Block::default().borders(Borders::BOTTOM));

        f.render_widget(header, area);
    }

    fn render_prices_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let prices = self.prices.lock().unwrap();
        let selected = *self.selected.lock().unwrap();
        let sort_column = *self.sort_column.lock().unwrap();
        let sort_ascending = *self.sort_ascending.lock().unwrap();

        let block = Block::default().borders(Borders::ALL).title("Live Prices");

        let inner_area = block.inner(area);
        f.render_widget(block, area);

        if inner_area.height < 3 || inner_area.width < 30 {
            return;
        }

        let mut symbols: Vec<String> = prices.keys().cloned().collect();

        symbols.sort_by(|a, b| {
            let a_data = prices.get(a);
            let b_data = prices.get(b);

            let ordering = match sort_column {
                0 => a.cmp(b),
                1 => a_data
                    .and_then(|d| d.prices.back())
                    .map(|p| p.price)
                    .partial_cmp(&b_data.and_then(|d| d.prices.back()).map(|p| p.price))
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => {
                    if let (Some(a_history), Some(b_history)) = (prices.get(a), prices.get(b)) {
                        if a_history.prices.len() >= 2 && b_history.prices.len() >= 2 {
                            let a_change = (a_history.prices.back().unwrap().price
                                - a_history.prices[a_history.prices.len() - 2].price)
                                / a_history.prices[a_history.prices.len() - 2].price;
                            let b_change = (b_history.prices.back().unwrap().price
                                - b_history.prices[b_history.prices.len() - 2].price)
                                / b_history.prices[b_history.prices.len() - 2].price;
                            a_change
                                .partial_cmp(&b_change)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
            };

            if sort_ascending {
                ordering
            } else {
                ordering.reverse()
            }
        });

        let rows = symbols.iter().enumerate().map(|(i, symbol)| {
            let is_selected = i == selected;
            let symbol_data = prices.get(symbol);
            let price = symbol_data
                .and_then(|d| d.prices.back())
                .map(|p| p.price)
                .unwrap_or(0.0);
            let volume = symbol_data
                .and_then(|d| d.prices.back())
                .map(|p| p.volume)
                .unwrap_or(0.0);
            let change = if let Some(data) = symbol_data {
                if data.prices.len() >= 2 {
                    ((data.prices.back().unwrap().price - data.prices[data.prices.len() - 2].price)
                        / data.prices[data.prices.len() - 2].price)
                        * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let change_color = if change < 0.0 {
                Color::Red
            } else {
                Color::Green
            };

            Row::new(vec![
                Cell::from(symbol.as_str()),
                Cell::from(Self::format_price(&price.to_string())),
                Cell::from(Span::styled(
                    Self::format_change(&change.to_string()),
                    Style::default().fg(change_color),
                )),
                Cell::from(Self::format_volume(&volume.to_string())),
            ])
                .style(if is_selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                })
        });

        let table = Table::new(rows)
            .header(
                Row::new(vec!["Pair", "Price", "24h Change", "Volume"])
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(12),
                Constraint::Length(15),
            ]);

        f.render_widget(table, inner_area);
    }

    fn render_alerts_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let alerts = self.alerts.lock().unwrap();
        let prices = self.prices.lock().unwrap();

        let block = Block::default().borders(Borders::ALL).title("Alerts");

        let inner_area = block.inner(area);
        f.render_widget(block, area);

        if inner_area.height < 3 || inner_area.width < 30 {
            return;
        }

        let rows = alerts.iter().map(|alert| {
            let price = prices
                .get(&alert.symbol)
                .and_then(|d| d.prices.back())
                .map(|p| p.price)
                .unwrap_or(0.0);

            let (condition_text, threshold) = match &alert.condition {
                AlertCondition::PriceAbove(t) => (">", t),
                AlertCondition::PriceBelow(t) => ("<", t),
                AlertCondition::ChangeAbove(t) => ("Δ>", t),
                AlertCondition::ChangeBelow(t) => ("Δ<", t),
                AlertCondition::IndicatorCross(ind, t) => (ind.as_str(), t),
            };

            let status = if alert.triggered {
                Span::styled("TRIGGERED", Style::default().fg(Color::Green))
            } else {
                Span::styled("PENDING", Style::default().fg(Color::Yellow))
            };

            Row::new(vec![
                Cell::from(alert.symbol.as_str()),
                Cell::from(format!("{} {:.2}", condition_text, threshold)),
                Cell::from(Self::format_price(&price.to_string())),
                Cell::from(status),
            ])
        });

        let table = Table::new(rows)
            .header(
                Row::new(vec!["Symbol", "Condition", "Current", "Status"])
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(15),
                Constraint::Length(12),
            ]);

        f.render_widget(table, inner_area);
    }

    fn render_portfolio_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let portfolio = self.portfolio.lock().unwrap();
        let prices = self.prices.lock().unwrap();

        let total_value = portfolio.current_value(&prices);
        let profit_loss = portfolio.profit_loss(&prices);
        let pct_change = (profit_loss / portfolio.initial_value) * 100.0;

        let summary = Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::raw("Total Value: "),
                Span::styled(
                    format!("${:.2}", total_value),
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
            Line::from(vec![
                Span::raw("Profit/Loss: "),
                Span::styled(
                    format!("${:.2} ({:.2}%)", profit_loss, pct_change),
                    Style::default().fg(if profit_loss >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
            ]),
            Line::from(vec![
                Span::raw("Available Cash: "),
                Span::styled(
                    format!("${:.2}", portfolio.cash),
                    Style::default().fg(Color::LightBlue),
                ),
            ]),
        ]))
            .block(Block::default().borders(Borders::ALL).title("Summary"));

        let rows = portfolio.holdings.iter().map(|(symbol, qty)| {
            let current_price = prices
                .get(symbol)
                .and_then(|d| d.prices.back())
                .map(|p| p.price)
                .unwrap_or(0.0);
            let value = qty * current_price;

            Row::new(vec![
                Cell::from(symbol.as_str()),
                Cell::from(format!("{:.4}", qty)),
                Cell::from(Self::format_price(&current_price.to_string())),
                Cell::from(format!("${:.2}", value)),
            ])
        });

        let holdings_table = Table::new(rows)
            .header(
                Row::new(vec!["Asset", "Amount", "Price", "Value"])
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .block(Block::default().borders(Borders::ALL).title("Holdings"))
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(15),
            ]);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(6), Constraint::Min(5)])
            .split(area);

        f.render_widget(summary, chunks[0]);
        f.render_widget(holdings_table, chunks[1]);
    }

    fn render_exchanges_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let message = Paragraph::new(Text::from(vec![
            Line::from("Multi-Exchange View"),
            Line::from(""),
            Line::from("Coming soon! Currently only Binance data is available."),
        ]))
            .block(Block::default().borders(Borders::ALL).title("Exchanges"));

        f.render_widget(message, area);
    }

    fn render_indicators_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let prices = self.prices.lock().unwrap();
        let selected = *self.selected.lock().unwrap();

        if let Some(symbol) = prices.keys().nth(selected) {
            if let Some(data) = prices.get(symbol) {
                let indicators = &data.indicators;
                let current_price = data.prices.back().map(|p| p.price).unwrap_or(0.0);

                let rows = vec![
                    Row::new(vec![
                        Cell::from("SMA (20)"),
                        Cell::from(Self::format_price(
                            &indicators.sma_20.unwrap_or(0.0).to_string(),
                        )),
                        Cell::from(
                            if indicators.sma_20.map_or(false, |sma| current_price > sma) {
                                "Price > SMA".to_string()
                            } else {
                                "Price ≤ SMA".to_string()
                            },
                        ),
                    ]),
                    Row::new(vec![
                        Cell::from("EMA (20)"),
                        Cell::from(Self::format_price(
                            &indicators.ema_20.unwrap_or(0.0).to_string(),
                        )),
                        Cell::from(
                            if indicators.ema_20.map_or(false, |ema| current_price > ema) {
                                "Price > EMA".to_string()
                            } else {
                                "Price ≤ EMA".to_string()
                            },
                        ),
                    ]),
                    Row::new(vec![
                        Cell::from("RSI (14)"),
                        Cell::from(format!("{:.2}", indicators.rsi_14.unwrap_or(0.0))),
                        Cell::from(if indicators.rsi_14.map_or(false, |rsi| rsi > 70.0) {
                            "Overbought (>70)".to_string()
                        } else if indicators.rsi_14.map_or(false, |rsi| rsi < 30.0) {
                            "Oversold (<30)".to_string()
                        } else {
                            "Neutral".to_string()
                        }),
                    ]),
                    Row::new(vec![
                        Cell::from("MACD"),
                        Cell::from(format!(
                            "{:.4} / {:.4}",
                            indicators.macd.unwrap_or(0.0),
                            indicators.macd_signal.unwrap_or(0.0)
                        )),
                        Cell::from(
                            if indicators
                                .macd
                                .zip(indicators.macd_signal)
                                .map_or(false, |(macd, signal)| macd > signal)
                            {
                                "Bullish (MACD > Signal)".to_string()
                            } else {
                                "Bearish (MACD ≤ Signal)".to_string()
                            },
                        ),
                    ]),
                    Row::new(vec![
                        Cell::from("Bollinger Bands"),
                        Cell::from(format!(
                            "U: {:.2} / L: {:.2}",
                            indicators.bollinger_upper.unwrap_or(0.0),
                            indicators.bollinger_lower.unwrap_or(0.0)
                        )),
                        Cell::from(
                            if indicators
                                .bollinger_upper
                                .map_or(false, |upper| current_price > upper)
                            {
                                "Above Upper Band".to_string()
                            } else if indicators
                                .bollinger_lower
                                .map_or(false, |lower| current_price < lower)
                            {
                                "Below Lower Band".to_string()
                            } else {
                                "Within Bands".to_string()
                            },
                        ),
                    ]),
                ];

                let table = Table::new(rows)
                    .header(
                        Row::new(vec!["Indicator", "Value", "Signal"])
                            .style(Style::default().add_modifier(Modifier::BOLD)),
                    )
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!("Indicators for {}", symbol)),
                    )
                    .widths(&[
                        Constraint::Length(15),
                        Constraint::Length(20),
                        Constraint::Length(25),
                    ]);

                f.render_widget(table, area);
                return;
            }
        }

        let message = Paragraph::new("No symbol selected or no data available")
            .block(Block::default().borders(Borders::ALL).title("Indicators"));
        f.render_widget(message, area);
    }

    fn render_chart_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let prices = self.prices.lock().unwrap();
        let selected = *self.selected.lock().unwrap();
        let timeframe = *self.selected_timeframe.lock().unwrap();

        if let Some(symbol) = prices.keys().nth(selected) {
            if let Some(data) = prices.get(symbol) {
                if data.prices.len() >= 2 {
                    let chart_area = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Length(3), Constraint::Min(5)])
                        .split(area);

                    let current_price = data.prices.back().unwrap().price;
                    let price_change = if data.prices.len() >= 2 {
                        let prev_price = data.prices[data.prices.len() - 2].price;
                        (current_price - prev_price) / prev_price * 100.0
                    } else {
                        0.0
                    };

                    let change_color = if price_change < 0.0 {
                        Color::Red
                    } else {
                        Color::Green
                    };

                    let header = Paragraph::new(Text::from(vec![
                        Line::from(vec![Span::styled(
                            format!("{} Price Chart ({})", symbol, timeframe.as_str()),
                            Style::default()
                                .fg(Color::LightCyan)
                                .add_modifier(Modifier::BOLD),
                        )]),
                        Line::from(vec![
                            Span::raw("Current: "),
                            Span::styled(
                                Self::format_price(&current_price.to_string()),
                                Style::default().fg(Color::Yellow),
                            ),
                            ratatui::prelude::Span::raw(" ("),
                            Span::styled(
                                Self::format_change(&price_change.to_string()),
                                Style::default().fg(change_color),
                            ),
                            Span::raw(")"),
                        ]),
                    ]))
                        .block(Block::default().borders(Borders::NONE));

                    f.render_widget(header, chart_area[0]);
                    self.render_line_chart(f, chart_area[1], symbol, data);
                    return;
                }
            }
        }

        let message = Paragraph::new("No symbol selected or insufficient data for chart")
            .block(Block::default().borders(Borders::ALL).title("Chart"));
        f.render_widget(message, area);
    }

    fn render_candles_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let mut prices = self.prices.lock().unwrap();
        let selected = *self.selected.lock().unwrap();
        let timeframe = *self.selected_timeframe.lock().unwrap();

        let symbol = prices.keys().nth(selected).map(|s| s.clone());

        if let Some(symbol) = symbol {
            if let Some(data) = prices.get_mut(&symbol) {
                data.update_candles(timeframe);

                if !data.candles.is_empty() {
                    let chart_area = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Length(3), Constraint::Min(5)])
                        .split(area);

                    let current_candle = data.candles.back().unwrap();
                    let header = Paragraph::new(Text::from(vec![
                        Line::from(vec![Span::styled(
                            format!("{} Candles ({})", symbol, timeframe.as_str()),
                            Style::default()
                                .fg(Color::LightCyan)
                                .add_modifier(Modifier::BOLD),
                        )]),
                        Line::from(vec![
                            Span::raw("O: "),
                            Span::styled(
                                Self::format_price(&current_candle.open.to_string()),
                                Style::default().fg(Color::Yellow),
                            ),
                            Span::raw(" H: "),
                            Span::styled(
                                Self::format_price(&current_candle.high.to_string()),
                                Style::default().fg(Color::Green),
                            ),
                            Span::raw(" L: "),
                            Span::styled(
                                Self::format_price(&current_candle.low.to_string()),
                                Style::default().fg(Color::Red),
                            ),
                            Span::raw(" C: "),
                            Span::styled(
                                Self::format_price(&current_candle.close.to_string()),
                                Style::default().fg(
                                    if current_candle.close >= current_candle.open {
                                        Color::Green
                                    } else {
                                        Color::Red
                                    },
                                ),
                            ),
                        ]),
                    ]))
                        .block(Block::default().borders(Borders::NONE));

                    f.render_widget(header, chart_area[0]);
                    self.render_candle_chart(f, chart_area[1], &symbol, data);
                    return;
                }
            }
        }

        let message = Paragraph::new("No symbol selected or insufficient data for candles")
            .block(Block::default().borders(Borders::ALL).title("Candles"));
        f.render_widget(message, area);
    }

    fn render_latency_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let latency_data = self.latency_data.lock().unwrap();
        let selected = *self.selected.lock().unwrap();
        let display_mode = *self.latency_display_mode.lock().unwrap();

        // Spliting area into table and chart sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Min(5)])
            .split(area);

        // Always render the table at the top
        self.render_latency_table(f, chunks[0], &latency_data, selected);

        // Render either line chart or sparkline based on display mode
        if let Some(symbol) = latency_data.keys().nth(selected) {
            if let Some(data) = latency_data.get(symbol) {
                if !data.is_empty() {
                    let max_latency = data.iter().map(|l| l.processing_time).max().unwrap_or(100) as f64;
                    let range = max_latency.max(10.0);

                    match display_mode {
                        LatencyDisplayMode::LineChart => {
                            self.render_latency_line_chart(f, chunks[1], symbol, data, range);
                        }
                        LatencyDisplayMode::Sparkline => {
                            self.render_sparkline_chart(f, chunks[1], symbol, data, range);
                        }
                    }
                    return;
                }
            }
        }

        // Fallback if no data
        let message = Paragraph::new("No latency data available for chart")
            .block(Block::default().borders(Borders::ALL).title("Chart"));
        f.render_widget(message, chunks[1]);
    }

    fn render_latency_table(
        &self,
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        latency_data: &HashMap<String, VecDeque<LatencyData>>,
        selected: usize,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Latency Metrics");

        let inner_area = block.inner(area);
        f.render_widget(block, area);

        if inner_area.height < 3 || inner_area.width < 30 {
            return;
        }

        let mut symbols: Vec<String> = latency_data.keys().cloned().collect();
        symbols.sort();

        let rows = symbols.iter().enumerate().map(|(i, symbol)| {
            let is_selected = i == selected;
            let data = latency_data.get(symbol);

            let avg_latency = data.map_or(0.0, |d| {
                if d.is_empty() {
                    0.0
                } else {
                    d.iter().map(|l| l.processing_time as f64).sum::<f64>() / d.len() as f64
                }
            });

            let max_latency = data.map_or(0.0, |d| {
                d.iter().map(|l| l.processing_time).max().unwrap_or(0) as f64
            });

            Row::new(vec![
                Cell::from(symbol.as_str()),
                Cell::from(format!("{:.1} ms", avg_latency)),
                Cell::from(format!("{:.1} ms", max_latency)),
                Cell::from(format!("{}", data.map_or(0, |d| d.len()))),
            ])
                .style(if is_selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                })
        });

        let table = Table::new(rows)
            .header(
                Row::new(vec!["Symbol", "Avg", "Max", "Samples"])
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
            ]);

        f.render_widget(table, inner_area);
    }
    fn render_latency_graph(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let latency_data = self.latency_data.lock().unwrap();
        let selected = *self.selected.lock().unwrap();

        // Spliting the area into header and chart
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)])
            .split(area);

        if let Some(symbol) = latency_data.keys().nth(selected) {
            if let Some(data) = latency_data.get(symbol) {
                if !data.is_empty() {
                    // Calculate stats
                    let current_latency = data.back().unwrap().processing_time;
                    let avg_latency = data.iter().map(|l| l.processing_time as f64).sum::<f64>()
                        / data.len() as f64;
                    let max_latency =
                        data.iter().map(|l| l.processing_time).max().unwrap_or(100) as f64;
                    let min_latency = 0.0;
                    let range = (max_latency - min_latency).max(10.0); // Minimum 10ms range for visibility

                    // Render header
                    let header = Paragraph::new(Text::from(vec![
                        Line::from(vec![Span::styled(
                            format!("{} Latency", symbol),
                            Style::default()
                                .fg(Color::LightCyan)
                                .add_modifier(Modifier::BOLD),
                        )]),
                        Line::from(vec![
                            Span::raw("Current: "),
                            Span::styled(
                                format!("{} ms", current_latency),
                                Style::default().fg(if current_latency > 100 {
                                    Color::Red
                                } else if current_latency > 50 {
                                    Color::Yellow
                                } else {
                                    Color::Green
                                }),
                            ),
                            Span::raw(" Avg: "),
                            Span::styled(
                                format!("{:.1} ms", avg_latency),
                                Style::default().fg(Color::Green),
                            ),
                            Span::raw(" Max: "),
                            Span::styled(
                                format!("{:.1} ms", max_latency),
                                Style::default().fg(Color::Red),
                            ),
                        ]),
                    ]))
                        .block(Block::default().borders(Borders::NONE));

                    f.render_widget(header, chunks[0]);

                    // Render the actual chart
                    self.render_latency_line_chart(f, chunks[1], symbol, data, range);
                    return;
                }
            }
        }

        // Fallback if no data
        let message = Paragraph::new("No latency data available")
            .block(Block::default().borders(Borders::ALL).title("Latency"));
        f.render_widget(message, area);
    }

    fn render_latency_line_chart(
        &self,
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        symbol: &str,
        data: &VecDeque<LatencyData>,
        range: f64,
    ) {
        // Early return for insufficient data
        if data.len() < 2 {
            let message = Paragraph::new("Insufficient data points for chart")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, area);
            return;
        }

        // 1. Enhanced chart block with better styling
        let chart_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Gray))
            .title(format!("{} Latency (ms)", symbol))
            .title_alignment(Alignment::Center);

        let inner_area = chart_block.inner(area);

        // Early return if area too small
        if inner_area.height < 3 || inner_area.width < 10 {
            return;
        }

        f.render_widget(chart_block, area);

        let height = inner_area.height as f64;
        let width = inner_area.width as f64;

        // 2. Dynamic step calculation for better rendering
        let step = (data.len() as f64 / width as f64).ceil() as usize;
        let step = step.max(1); // Ensure at least 1

        // 3. Improved Y-axis labels
        let y_axis_labels = vec![
            (0, format!("{:.0}", range)),                           // Top
            (inner_area.height / 2, format!("{:.0}", range / 2.0)), // Middle
            (inner_area.height - 1, "0".to_string()),               // Bottom
        ];

        for (y, label) in y_axis_labels {
            let label_widget = Paragraph::new(label).style(Style::default().fg(Color::DarkGray));
            f.render_widget(
                label_widget,
                Rect::new(inner_area.right() - 5, inner_area.y + y, 5, 1),
            );
        }

        // 4. Threshold lines for visual reference
        let warning_threshold = (height * 0.7) as u16; // 70% of height
        let critical_threshold = (height * 0.9) as u16; // 90% of height

        // Draw threshold lines
        for (y, color, label) in [
            (warning_threshold, Color::Yellow, "Warning"),
            (critical_threshold, Color::Red, "Critical"),
        ] {
            let line = Line::from(vec![
                Span::styled(
                    "─".repeat(inner_area.width as usize),
                    Style::default().fg(color),
                ),
                Span::styled(
                    label,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]);

            f.render_widget(
                Paragraph::new(line),
                Rect::new(inner_area.x, inner_area.y + y, inner_area.width, 1),
            );
        }

        // 5. More efficient point calculation and rendering
        let points: Vec<(u16, u16)> = data
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, l)| {
                let x = inner_area.x + (i as f64 / data.len() as f64 * width as f64).round() as u16;
                let y = inner_area.y
                    + (height - 1.0 - (l.processing_time as f64 / range * (height - 2.0))) as u16;
                (x, y)
            })
            .collect();

        // 6. Use Sparkline for smoother rendering
        if let Some(sparkline) = self.create_sparkline(&points, range) {
            f.render_widget(sparkline, inner_area);
        }

        // 7. Enhanced current value marker
        if let Some(last) = data.back() {
            let default_point = (inner_area.right(), inner_area.bottom());
            let last_point = points.last().unwrap_or(&default_point);
            let marker =
                Paragraph::new("◉").style(Style::default().fg(match last.processing_time {
                    t if t > 100 => Color::Red,
                    t if t > 50 => Color::Yellow,
                    _ => Color::Green,
                }));
            f.render_widget(marker, Rect::new(last_point.0, last_point.1, 1, 1));

            // Add value label
            let value_label = Paragraph::new(format!("{}ms", last.processing_time))
                .style(Style::default().fg(Color::White));
            f.render_widget(
                value_label,
                Rect::new(last_point.0.saturating_sub(5), last_point.1, 10, 1),
            );
        }
    }

    // Helper function for sparkline creation
    fn create_sparkline(&self, points: &[(u16, u16)], max_value: f64) -> Option<Sparkline<'static>> {
        if points.is_empty() {
            return None;
        }

        // Create the data vector with proper name
        let data: Vec<u64> = points.iter()
            .map(|(_, y)| *y as u64)
            .collect();

        Some(
            Sparkline::default()
                .data(data.leak()) // Convert to 'static lifetime
                .max(max_value as u64)
                .style(Style::default().fg(Color::LightBlue))
                .bar_set(ratatui::symbols::bar::NINE_LEVELS)
        )
    }
    fn render_sparkline_chart(
        &self,
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        symbol: &str,
        data: &VecDeque<LatencyData>,
        range: f64,
    ) {
        let chart_block = Block::default()
            .borders(Borders::ALL)
            .title(format!("{} Latency (Sparkline)", symbol));

        let inner_area = chart_block.inner(area);
        f.render_widget(chart_block, area);

        // Convert latency data to points
        let points: Vec<(u16, u16)> = data
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let x = (i as f64 / data.len() as f64 * inner_area.width as f64).round() as u16;
                let y =
                    (l.processing_time as f64 / range * inner_area.height as f64).round() as u16;
                (x, y)
            })
            .collect();

        if let Some(sparkline) = self.create_sparkline(&points, range) {
            f.render_widget(sparkline, inner_area);
        }
    }
    fn render_main_content(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        match *self.current_view.lock().unwrap() {
            DashboardView::Prices => self.render_prices_view(f, area),
            DashboardView::Alerts => self.render_alerts_view(f, area),
            DashboardView::Portfolio => self.render_portfolio_view(f, area),
            DashboardView::Exchanges => self.render_exchanges_view(f, area),
            DashboardView::Indicators => self.render_indicators_view(f, area),
            DashboardView::Chart => self.render_chart_view(f, area),
            DashboardView::Candles => self.render_candles_view(f, area),
            DashboardView::Latency => self.render_latency_view(f, area),
        }
    }

    fn render_footer(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let current_view = *self.current_view.lock().unwrap();

        let controls = match current_view {
            DashboardView::Prices => vec![
                Span::raw("Controls: "),
                Span::styled("↑/↓", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Navigate  "),
                Span::styled("s", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Sort  "),
                Span::styled("a", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Alerts  "),
                Span::styled("p", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Portfolio  "),
                Span::styled("e", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Exchanges  "),
                Span::styled("i", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Indicators  "),
                Span::styled("c", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Chart  "),
                Span::styled("k", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Candles  "),
                Span::styled("b", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Buy  "),
                Span::styled("x", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Sell  "),
                Span::styled("t", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Timeframe  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
                Span::styled("l", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Latency  "),
            ],
            DashboardView::Chart | DashboardView::Candles => vec![
                Span::raw("Controls: "),
                Span::styled("t", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Timeframe  "),
                Span::styled("v", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Prices  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
            DashboardView::Latency => {
                let display_mode = *self.latency_display_mode.lock().unwrap();
                let zoom = *self.latency_zoom.lock().unwrap();

                vec![
                    Span::raw("Mode: "),
                    Span::styled(
                        format!("{:?} ", display_mode),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("Zoom: "),
                    ratatui::prelude::Span::styled(format!("{} ", zoom), Style::default().fg(Color::Yellow)),
                    Span::raw("Controls: "),
                    Span::styled("←/→", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" Navigate "),
                    Span::styled("t", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" Toggle "),
                    Span::styled("h", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" Thresholds "),
                    Span::styled("z", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" Zoom "),
                    Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(" Quit"),
                ]
            }
            _ => vec![
                Span::raw("Controls: "),
                Span::styled("v", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Prices  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
        };

        let footer = Paragraph::new(Line::from(controls))
            .style(Style::default().fg(Color::Gray))
            .block(Block::default().borders(Borders::TOP));

        f.render_widget(footer, area);
    }


    fn render_line_chart(
        &self,
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        symbol: &str,
        data: &SymbolData,
    ) {
        if data.prices.len() < 2 {
            let message = Paragraph::new("Insufficient data for chart")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, area);
            return;
        }

        let min_price = data
            .prices
            .iter()
            .map(|p| p.price)
            .fold(f64::INFINITY, f64::min);
        let max_price = data.prices.iter().map(|p| p.price).fold(0.0, f64::max);
        let price_range = max_price - min_price;

        if price_range <= 0.0 {
            let message = Paragraph::new("No price variation to display")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, area);
            return;
        }

        let chart_block = Block::default()
            .borders(Borders::ALL)
            .title(format!("{} Price History", symbol));

        let inner_area = chart_block.inner(area);
        if inner_area.height < 3 || inner_area.width < 10 {
            return;
        }
        f.render_widget(chart_block, area);

        let height = inner_area.height as f64;
        let width = inner_area.width as f64;
        let step = (data.prices.len() as f64 / width).ceil() as usize;
        let points: Vec<(f64, f64)> = data
            .prices
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, p)| {
                let x = (i as f64 / step as f64).min(width - 1.0);
                let y = height - 1.0 - ((p.price - min_price) / price_range * (height - 2.0));
                (x, y)
            })
            .collect();

        for window in points.windows(2) {
            if let [(x1, y1), (x2, y2)] = window {
                let start_x = inner_area.x + x1.round() as u16;
                let start_y = inner_area.y + y1.round() as u16;
                let end_x = inner_area.x + x2.round() as u16;
                let end_y = inner_area.y + y2.round() as u16;

                if start_x == end_x && start_y == end_y {
                    let dot = Paragraph::new("◉").style(Style::default().fg(Color::LightBlue));
                    f.render_widget(dot, Rect::new(start_x, start_y, 1, 1));
                } else {
                    let mut x = start_x;
                    let mut y = start_y;
                    let dx = end_x as i16 - start_x as i16;
                    let dy = end_y as i16 - start_y as i16;
                    let step = dx.abs().max(dy.abs());

                    for _ in 0..=step {
                        let dot = Paragraph::new("▪").style(Style::default().fg(Color::LightBlue));
                        f.render_widget(dot, Rect::new(x, y, 1, 1));
                        x = (x as i16 + dx / step) as u16;
                        y = (y as i16 + dy / step) as u16;
                    }
                }
            }
        }

        let top_label =
            Paragraph::new(format!("{:.2}", max_price)).style(Style::default().fg(Color::Gray));
        let bottom_label =
            Paragraph::new(format!("{:.2}", min_price)).style(Style::default().fg(Color::Gray));

        f.render_widget(
            top_label,
            Rect::new(inner_area.right() - 8, inner_area.y, 8, 1),
        );
        f.render_widget(
            bottom_label,
            Rect::new(inner_area.right() - 8, inner_area.bottom() - 1, 8, 1),
        );
    }
    fn render_candle_chart(
        &self,
        f: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        symbol: &str,
        data: &SymbolData,
    ) {
        if data.candles.len() < 2 {
            let message = Paragraph::new("Insufficient data for candles")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, area);
            return;
        }

        let min_price = data
            .candles
            .iter()
            .map(|c| c.low)
            .fold(f64::INFINITY, f64::min);
        let max_price = data.candles.iter().map(|c| c.high).fold(0.0, f64::max);
        let price_range = max_price - min_price;

        if price_range <= 0.0 {
            let message = Paragraph::new("No price variation to display")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(message, area);
            return;
        }

        let chart_block = Block::default()
            .borders(Borders::ALL)
            .title(format!("{} Candles", symbol));

        let inner_area = chart_block.inner(area);
        if inner_area.height < 3 || inner_area.width < 10 {
            return;
        }
        f.render_widget(chart_block, area);

        let candle_width = (inner_area.width as f32 / data.candles.len() as f32).max(1.0) as u16;
        let candle_spacing = 1;

        for (i, candle) in data.candles.iter().enumerate() {
            let x = inner_area.x + (i as u16 * (candle_width + candle_spacing));
            if x >= inner_area.right() {
                break;
            }

            let open_y = inner_area.y + (inner_area.height - 1)
                - (((candle.open - min_price) / price_range) * (inner_area.height - 1) as f64).round()
                as u16;
            let close_y = inner_area.y + (inner_area.height - 1)
                - (((candle.close - min_price) / price_range) * (inner_area.height - 1) as f64).round()
                as u16;
            let high_y = inner_area.y + (inner_area.height - 1)
                - (((candle.high - min_price) / price_range) * (inner_area.height - 1) as f64).round()
                as u16;
            let low_y = inner_area.y + (inner_area.height - 1)
                - (((candle.low - min_price) / price_range) * (inner_area.height - 1) as f64).round()
                as u16;

            let is_bullish = candle.close >= candle.open;
            let color = if is_bullish { Color::Green } else { Color::Red };

            // Draw wick (high to low)
            for y in low_y..=high_y {
                let wick = Paragraph::new("│").style(Style::default().fg(color));
                f.render_widget(wick, Rect::new(x + candle_width / 2, y, 1, 1));
            }

            // Draw body
            let (top, bottom) = if is_bullish {
                (close_y, open_y)
            } else {
                (open_y, close_y)
            };

            for y in bottom..=top {
                for w in 0..candle_width {
                    let body = Paragraph::new("█").style(Style::default().fg(color));
                    f.render_widget(body, Rect::new(x + w, y, 1, 1));
                }
            }
        }

        let top_label =
            Paragraph::new(format!("{:.2}", max_price)).style(Style::default().fg(Color::Gray));
        let bottom_label =
            Paragraph::new(format!("{:.2}", min_price)).style(Style::default().fg(Color::Gray));

        f.render_widget(
            top_label,
            Rect::new(inner_area.right() - 8, inner_area.y, 8, 1),
        );
        f.render_widget(
            bottom_label,
            Rect::new(inner_area.right() - 8, inner_area.bottom() - 1, 8, 1),
        );
    }
    fn format_price(price: &str) -> String {
        price.parse::<f64>().map_or_else(
            |_| format!("{:>10}", price),
            |num| {
                if num > 1000.0 {
                    format!("${:>10.2}", num)
                } else {
                    format!("${:>10.4}", num)
                }
            },
        )
    }

    fn format_change(change: &str) -> String {
        change
            .parse::<f64>()
            .map_or_else(|_| format!("{:>7}", change), |c| format!("{:>7.2}%", c),
            )
    }

    fn format_volume(volume: &str) -> String {
        volume.parse::<f64>().map_or_else(
            |_| format!("{:>15}", volume),
            |v| {
                if v > 1_000_000.0 {
                    format!("{:>10.2}M", v / 1_000_000.0)
                } else if v > 1_000.0 {
                    format!("{:>10.2}K", v / 1_000.0)
                } else {
                    format!("{:>10.2}", v)
                }
            },
        )
    }
}
