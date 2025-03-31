use chrono::Local;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
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
    pub indicators: IndicatorData,
}

impl SymbolData {
    pub fn new() -> Self {
        Self {
            prices: VecDeque::new(),
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
        self.holdings
            .iter()
            .fold(0.0, |acc, (symbol, qty)| {
                prices
                    .get(symbol)
                    .and_then(|data| data.prices.back())
                    .map(|price_data| price_data.price * qty)
                    .unwrap_or(0.0)
                    + acc
            })
            + self.cash
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
        }
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
            KeyCode::Char('v') => *self.current_view.lock().unwrap() = DashboardView::Prices,
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
            KeyCode::Char('t') => {
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
            _ => {}
        }
        Ok(())
    }

    async fn update_price(
        &self,
        symbol: &str,
        price: &str,
        change: &str,
    ) -> Result<(), DynError> {
        
        if price.trim().is_empty() {
            log::warn!("Empty price received for {}", symbol);
            return Ok(());
        }

        // Parse price with error handling
        let price_num = match price.parse::<f64>() {
            Ok(num) => num,
            Err(e) => {
                log::error!("Failed to parse price '{}' for {}: {}", price, symbol, e);
                return Ok(()); // Skip this update but keep running
            }
        };

        
        let change_num = change.parse::<f64>().unwrap_or(0.0);

        let timestamp = Local::now().timestamp();

        let price_data = PriceData {
            price: price_num,
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
                AlertCondition::IndicatorCross(indicator, threshold) => {
                    match indicator.as_str() {
                        "RSI" => symbol_data.indicators.rsi_14.map_or(false, |rsi| rsi > *threshold),
                        "MACD" => symbol_data
                            .indicators
                            .macd
                            .zip(symbol_data.indicators.macd_signal)
                            .map_or(false, |(macd, signal)| macd > signal),
                        "BB" => price_data.price > symbol_data.indicators.bollinger_upper.unwrap_or(0.0)
                            || price_data.price < symbol_data.indicators.bollinger_lower.unwrap_or(0.0),
                        _ => false,
                    }
                }
            };
            alert.triggered = should_trigger;
        }
        Ok(())
    }

    fn render_header(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let portfolio = self.portfolio.lock().unwrap();
        let prices = self.prices.lock().unwrap();

        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "CRYPTOWATCH DASHBOARD",
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!(
                    "Last update: {} | Portfolio: ${:.2} | P/L: {:.2}% | Timeframe: {:?}",
                    Local::now().format("%H:%M:%S"),
                    portfolio.current_value(&prices),
                    portfolio.profit_loss(&prices) / portfolio.initial_value * 100.0,
                    *self.selected_timeframe.lock().unwrap()
                ),
                Style::default().fg(Color::Gray),
            )),
        ]))
            .block(Block::default().borders(Borders::BOTTOM));

        f.render_widget(header, area);
    }

    fn render_main_content(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        match *self.current_view.lock().unwrap() {
            DashboardView::Prices => self.render_prices_view(f, area),
            DashboardView::Alerts => self.render_alerts_view(f, area),
            DashboardView::Portfolio => self.render_portfolio_view(f, area),
            DashboardView::Exchanges => self.render_exchanges_view(f, area),
            DashboardView::Indicators => self.render_indicators_view(f, area),
            DashboardView::Chart => self.render_chart_view(f, area),
        }
    }

    fn render_prices_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let prices = self.prices.lock().unwrap();
        let selected = *self.selected.lock().unwrap();
        let sort_column = *self.sort_column.lock().unwrap();
        let sort_ascending = *self.sort_ascending.lock().unwrap();

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
                            a_change.partial_cmp(&b_change).unwrap_or(std::cmp::Ordering::Equal)
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
            let price = symbol_data.and_then(|d| d.prices.back()).map(|p| p.price).unwrap_or(0.0);
            let volume = symbol_data.and_then(|d| d.prices.back()).map(|p| p.volume).unwrap_or(0.0);
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
                Cell::from(format_price(&price.to_string())),
                Cell::from(Span::styled(
                    format_change(&change.to_string()),
                    Style::default().fg(change_color),
                )),
                Cell::from(format_volume(&volume.to_string())),
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
            .block(Block::default().borders(Borders::ALL).title("Live Prices"))
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(12),
                Constraint::Length(15),
            ]);

        f.render_widget(table, area);
    }

    fn render_alerts_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let alerts = self.alerts.lock().unwrap();
        let prices = self.prices.lock().unwrap();

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
                Cell::from(format_price(&price.to_string())),
                Cell::from(status),
            ])
        });

        let table = Table::new(rows)
            .header(
                Row::new(vec!["Symbol", "Condition", "Current", "Status"])
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .block(Block::default().borders(Borders::ALL).title("Alerts"))
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(15),
                Constraint::Length(12),
            ]);

        f.render_widget(table, area);
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
                Cell::from(format_price(&current_price.to_string())),
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
                        Cell::from(format_price(
                            &indicators.sma_20.unwrap_or(0.0).to_string(),
                        )),
                        Cell::from(if indicators.sma_20.map_or(false, |sma| current_price > sma) {
                            "Price > SMA".to_string()
                        } else {
                            "Price ≤ SMA".to_string()
                        }),
                    ]),
                    Row::new(vec![
                        Cell::from("EMA (20)"),
                        Cell::from(format_price(
                            &indicators.ema_20.unwrap_or(0.0).to_string(),
                        )),
                        Cell::from(if indicators.ema_20.map_or(false, |ema| current_price > ema) {
                            "Price > EMA".to_string()
                        } else {
                            "Price ≤ EMA".to_string()
                        }),
                    ]),
                    Row::new(vec![
                        Cell::from("RSI (14)"),
                        Cell::from(format!(
                            "{:.2}",
                            indicators.rsi_14.unwrap_or(0.0)
                        )),
                        Cell::from(
                            if indicators.rsi_14.map_or(false, |rsi| rsi > 70.0) {
                                "Overbought (>70)".to_string()
                            } else if indicators.rsi_14.map_or(false, |rsi| rsi < 30.0) {
                                "Oversold (<30)".to_string()
                            } else {
                                "Neutral".to_string()
                            },
                        ),
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
                            if indicators.bollinger_upper.map_or(false, |upper| {
                                current_price > upper
                            }) {
                                "Above Upper Band".to_string()
                            } else if indicators.bollinger_lower.map_or(false, |lower| {
                                current_price < lower
                            }) {
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

        if let Some(symbol) = prices.keys().nth(selected) {
            if let Some(data) = prices.get(symbol) {
                if data.prices.len() >= 2 {
                    let chart = Paragraph::new(Text::from(vec![
                        Line::from(format!("Price Chart for {}", symbol)),
                        Line::from(""),
                        Line::from("Coming soon! This will show a proper price chart."),
                        Line::from(""),
                        Line::from(format!(
                            "Current Price: {}",
                            format_price(&data.prices.back().unwrap().price.to_string())
                        )),
                    ]))
                        .block(Block::default().borders(Borders::ALL).title("Chart"));

                    f.render_widget(chart, area);
                    return;
                }
            }
        }

        let message = Paragraph::new("No symbol selected or insufficient data for chart")
            .block(Block::default().borders(Borders::ALL).title("Chart"));
        f.render_widget(message, area);
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
                Span::styled("b", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Buy  "),
                Span::styled("x", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Sell  "),
                Span::styled("t", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Timeframe  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
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
    change.parse::<f64>().map_or_else(
        |_| format!("{:>7}", change),
        |c| format!("{:>7.2}%", c),
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
