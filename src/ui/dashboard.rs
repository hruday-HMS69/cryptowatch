use chrono::Local;
use crossterm::{
    event::{self, Event, KeyCode},
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
use std::collections::HashMap;
use std::io;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DashboardView {
    Prices,
    Alerts,
    Portfolio,
    Exchanges,
    Indicators,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    PriceAbove(f64),
    PriceBelow(f64),
    ChangeAbove(f64),
    ChangeBelow(f64),
    IndicatorCross { indicator: String, value: f64, direction: CrossDirection },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossDirection {
    Above,
    Below,
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub symbol: String,
    pub condition: AlertCondition,
    pub triggered: bool,
    pub active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Exchange {
    Binance,
    Coinbase,
    Kraken,
}

#[derive(Debug, Clone)]
struct PriceData {
    pub price: f64,
    pub volume: f64,
    pub timestamp: i64,
    pub exchange: Exchange,
}

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub holdings: HashMap<String, f64>,
    pub initial_value: f64,
    pub cash: f64,
}

#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    pub sma: HashMap<usize, Vec<f64>>,
    pub rsi: HashMap<usize, Vec<f64>>,
    pub macd: HashMap<(usize, usize, usize), MacdData>,
}

#[derive(Debug, Clone)]
pub struct MacdData {
    pub macd_line: Vec<f64>,
    pub signal_line: Vec<f64>,
    pub histogram: Vec<f64>,
}

impl TechnicalIndicators {
    pub fn new() -> Self {
        Self {
            sma: HashMap::new(),
            rsi: HashMap::new(),
            macd: HashMap::new(),
        }
    }
}

impl Portfolio {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            holdings: HashMap::new(),
            initial_value: initial_cash,
            cash: initial_cash,
        }
    }

    pub fn current_value(&self, prices: &HashMap<String, Vec<PriceData>>) -> f64 {
        self.holdings.iter().fold(0.0, |acc, (symbol, qty)| {
            prices.get(symbol)
                .and_then(|prices| prices.first())
                .map(|price_data| price_data.price * qty)
                .unwrap_or(0.0) + acc
        }) + self.cash
    }

    pub fn profit_loss(&self, prices: &HashMap<String, Vec<PriceData>>) -> f64 {
        self.current_value(prices) - self.initial_value
    }
}

pub struct Dashboard {
    pub prices: HashMap<String, Vec<PriceData>>,
    pub selected: usize,
    pub sort_column: usize,
    pub sort_ascending: bool,
    pub alerts: Vec<Alert>,
    pub portfolio: Portfolio,
    pub current_view: DashboardView,
    pub price_history_length: usize,
    pub indicators: TechnicalIndicators,
    pub indicator_periods: Vec<usize>,
    pub selected_indicator: usize,
    pub show_indicators: bool,
    pub initialized: bool,
}

impl Dashboard {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            prices: HashMap::new(),
            selected: 0,
            sort_column: 0,
            sort_ascending: false,
            alerts: vec![
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
            ],
            portfolio: Portfolio::new(initial_cash),
            current_view: DashboardView::Prices,
            price_history_length: 50,
            indicators: TechnicalIndicators::new(),
            indicator_periods: vec![7, 14, 21, 50, 200],
            selected_indicator: 0,
            show_indicators: false,
            initialized: false,
        }
    }

    pub async fn run(
        mut receiver: mpsc::Receiver<(String, String, String)>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        let mut dashboard = Self::new(10000.0);
        let mut last_render_time = Local::now();

        loop {
            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('p') => dashboard.current_view = DashboardView::Portfolio,
                        KeyCode::Char('a') => dashboard.current_view = DashboardView::Alerts,
                        KeyCode::Char('e') => dashboard.current_view = DashboardView::Exchanges,
                        KeyCode::Char('v') => dashboard.current_view = DashboardView::Prices,
                        KeyCode::Char('i') => dashboard.current_view = DashboardView::Indicators,
                        KeyCode::Char('t') => dashboard.show_indicators = !dashboard.show_indicators,
                        KeyCode::Up => {
                            if dashboard.current_view == DashboardView::Indicators {
                                dashboard.selected_indicator = dashboard.selected_indicator.saturating_sub(1);
                            } else {
                                dashboard.selected = dashboard.selected.saturating_sub(1);
                            }
                        },
                        KeyCode::Down => {
                            if dashboard.current_view == DashboardView::Indicators {
                                dashboard.selected_indicator = (dashboard.selected_indicator + 1)
                                    .min(dashboard.indicator_periods.len() - 1);
                            } else {
                                dashboard.selected = dashboard.selected.saturating_add(1);
                            }
                        },
                        KeyCode::Char('s') => {
                            dashboard.sort_column = (dashboard.sort_column + 1) % 3;
                            dashboard.sort_ascending = !dashboard.sort_ascending;
                        }
                        _ => {}
                    }
                }
            }

            while let Ok((symbol, price, change)) = receiver.try_recv() {
                dashboard.update_price(&symbol, &price, &change);
                if dashboard.prices.values().next().map(|v| v.len()).unwrap_or(0) > 14 {
                    dashboard.initialized = true;
                    dashboard.check_alerts(&symbol);
                }
            }

            // Throttle rendering to avoid high CPU usage
            let now = Local::now();
            if (now - last_render_time).num_milliseconds() > 100 {
                terminal.draw(|f| dashboard.render(f))?;
                last_render_time = now;
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    fn update_price(&mut self, symbol: &str, price: &str, change: &str) {
        let price_num = price.parse::<f64>().unwrap_or(0.0);
        let _change_num = change.parse::<f64>().unwrap_or(0.0);
        let timestamp = Local::now().timestamp();

        let price_data = PriceData {
            price: price_num,
            volume: 0.0,
            timestamp,
            exchange: Exchange::Binance,
        };

        self.prices
            .entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .push(price_data);

        if let Some(prices) = self.prices.get_mut(symbol) {
            if prices.len() > self.price_history_length {
                prices.remove(0);
            }
        }

        if self.initialized {
            self.calculate_indicators();
        }
    }

    fn calculate_indicators(&mut self) {
        for (_symbol, prices) in &self.prices {
            if prices.len() < 15 { // Minimum data points needed for calculations
                continue;
            }

            let price_values: Vec<f64> = prices.iter().map(|p| p.price).collect();

            for &period in &self.indicator_periods {
                let sma = self.calculate_sma(&price_values, period);
                self.indicators.sma.insert(period, sma);
            }

            for &period in &self.indicator_periods {
                let rsi = self.calculate_rsi(&price_values, period);
                self.indicators.rsi.insert(period, rsi);
            }

            let macd_data = self.calculate_macd(&price_values, 12, 26, 9);
            self.indicators.macd.insert((12, 26, 9), macd_data);
        }
    }

    fn calculate_sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![0.0; prices.len()];
        }

        let mut sma = Vec::with_capacity(prices.len());
        for i in 0..prices.len() {
            if i < period - 1 {
                sma.push(0.0);
            } else {
                let sum: f64 = prices[i - period + 1..=i].iter().sum();
                sma.push(sum / period as f64);
            }
        }
        sma
    }

    fn calculate_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![0.0; prices.len()];
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(change.abs());
            }
        }

        let mut avg_gain = gains[0..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[0..period].iter().sum::<f64>() / period as f64;

        let mut rsi = vec![0.0; period];
        if avg_loss == 0.0 {
            rsi.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            rsi.push(100.0 - (100.0 / (1.0 + rs)));
        }

        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss == 0.0 {
                rsi.push(100.0);
            } else {
                let rs = avg_gain / avg_loss;
                rsi.push(100.0 - (100.0 / (1.0 + rs)));
            }
        }

        let mut full_rsi = vec![0.0; prices.len() - rsi.len()];
        full_rsi.extend(rsi);
        full_rsi
    }

    fn calculate_macd(&self, prices: &[f64], fast: usize, slow: usize, signal: usize) -> MacdData {
        let fast_ema = self.calculate_ema(prices, fast);
        let slow_ema = self.calculate_ema(prices, slow);

        let macd_line: Vec<f64> = fast_ema.iter().zip(&slow_ema)
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = self.calculate_ema(&macd_line[slow - fast..], signal);

        let histogram: Vec<f64> = macd_line[slow - fast + signal - 1..].iter()
            .zip(&signal_line)
            .map(|(m, s)| m - s)
            .collect();

        let pad_len = prices.len() - macd_line.len();
        let mut full_macd_line = vec![0.0; pad_len];
        full_macd_line.extend(macd_line);

        let pad_len = prices.len() - signal_line.len();
        let mut full_signal_line = vec![0.0; pad_len];
        full_signal_line.extend(signal_line);

        let pad_len = prices.len() - histogram.len();
        let mut full_histogram = vec![0.0; pad_len];
        full_histogram.extend(histogram);

        MacdData {
            macd_line: full_macd_line,
            signal_line: full_signal_line,
            histogram: full_histogram,
        }
    }

    fn calculate_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![0.0; prices.len()];
        }

        let mut ema = Vec::with_capacity(prices.len());
        let multiplier = 2.0 / (period as f64 + 1.0);

        let first_sma: f64 = prices[0..period].iter().sum::<f64>() / period as f64;
        ema.push(first_sma);

        for i in period..prices.len() {
            let current_ema = (prices[i] - ema.last().unwrap()) * multiplier + ema.last().unwrap();
            ema.push(current_ema);
        }

        let mut full_ema = vec![0.0; prices.len() - ema.len()];
        full_ema.extend(ema);
        full_ema
    }

    fn check_alerts(&mut self, symbol: &str) {
        if let Some(price_data) = self.prices.get(symbol).and_then(|p| p.last()) {
            for alert in &mut self.alerts {
                if alert.symbol == *symbol && alert.active {
                    let should_trigger = match &alert.condition {
                        AlertCondition::PriceAbove(threshold) => price_data.price > *threshold,
                        AlertCondition::PriceBelow(threshold) => price_data.price < *threshold,
                        AlertCondition::ChangeAbove(threshold) => {
                            if let Some(prices) = self.prices.get(symbol) {
                                if prices.len() >= 2 {
                                    let prev_price = prices[prices.len() - 2].price;
                                    let change = (price_data.price - prev_price) / prev_price * 100.0;
                                    change > *threshold
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        AlertCondition::ChangeBelow(threshold) => {
                            if let Some(prices) = self.prices.get(symbol) {
                                if prices.len() >= 2 {
                                    let prev_price = prices[prices.len() - 2].price;
                                    let change = (price_data.price - prev_price) / prev_price * 100.0;
                                    change < *threshold
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        AlertCondition::IndicatorCross { indicator, value, direction } => {
                            let indicator_value = match indicator.to_lowercase().as_str() {
                                "sma" => self.indicators.sma.get(&14).and_then(|v| v.last()).copied().unwrap_or(0.0),
                                "rsi" => self.indicators.rsi.get(&14).and_then(|v| v.last()).copied().unwrap_or(0.0),
                                "macd" => self.indicators.macd.get(&(12, 26, 9)).and_then(|m| m.macd_line.last()).copied().unwrap_or(0.0),
                                _ => 0.0,
                            };
                            match direction {
                                CrossDirection::Above => indicator_value > *value,
                                CrossDirection::Below => indicator_value < *value,
                            }
                        }
                    };
                    alert.triggered = should_trigger;
                }
            }
        }
    }

    fn render(&mut self, f: &mut Frame<CrosstermBackend<io::Stdout>>) {
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
    }

    fn render_header(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "CRYPTOWATCH DASHBOARD",
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!(
                    "Last update: {} | Portfolio: ${:.2} | P/L: {:.2}% | Indicators: {}",
                    Local::now().format("%H:%M:%S"),
                    self.portfolio.current_value(&self.prices),
                    self.portfolio.profit_loss(&self.prices) / self.portfolio.initial_value * 100.0,
                    if self.show_indicators { "ON" } else { "OFF" }
                ),
                Style::default().fg(Color::Gray),
            )),
        ]))
            .block(Block::default().borders(Borders::BOTTOM));

        f.render_widget(header, area);
    }

    fn render_main_content(&mut self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        if !self.initialized {
            let loading = Paragraph::new("Loading data... (waiting for enough price history)")
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(loading, area);
            return;
        }

        match self.current_view {
            DashboardView::Prices => self.render_prices_view(f, area),
            DashboardView::Alerts => self.render_alerts_view(f, area),
            DashboardView::Portfolio => self.render_portfolio_view(f, area),
            DashboardView::Exchanges => self.render_exchanges_view(f, area),
            DashboardView::Indicators => self.render_indicators_view(f, area),
        }
    }

    fn render_prices_view(&mut self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let mut symbols: Vec<&String> = self.prices.keys().collect();

        symbols.sort_by(|a, b| {
            let a_data = self.prices.get(*a).and_then(|p| p.last());
            let b_data = self.prices.get(*b).and_then(|p| p.last());

            let ordering = match self.sort_column {
                0 => a.cmp(b),
                1 => a_data.map(|a| a.price)
                    .partial_cmp(&b_data.map(|b| b.price))
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => {
                    if let (Some(a_history), Some(b_history)) = (self.prices.get(*a), self.prices.get(*b)) {
                        if a_history.len() >= 2 && b_history.len() >= 2 {
                            let a_change = (a_history.last().unwrap().price - a_history[a_history.len() - 2].price)
                                / a_history[a_history.len() - 2].price;
                            let b_change = (b_history.last().unwrap().price - b_history[b_history.len() - 2].price)
                                / b_history[b_history.len() - 2].price;
                            a_change.partial_cmp(&b_change).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
            };

            if self.sort_ascending { ordering } else { ordering.reverse() }
        });

        let rows = symbols.iter().enumerate().map(|(i, symbol)| {
            let is_selected = i == self.selected;
            let price_data = self.prices.get(*symbol).and_then(|p| p.last());
            let price = price_data.map(|p| p.price).unwrap_or(0.0);
            let change = if let Some(history) = self.prices.get(*symbol) {
                if history.len() >= 2 {
                    ((history.last().unwrap().price - history[history.len() - 2].price)
                        / history[history.len() - 2].price) * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let change_color = if change < 0.0 { Color::Red } else { Color::Green };

            let sma_value = self.indicators.sma.get(&14).and_then(|v| v.last()).copied().unwrap_or(0.0);
            let rsi_value = self.indicators.rsi.get(&14).and_then(|v| v.last()).copied().unwrap_or(0.0);
            let macd_value = self.indicators.macd.get(&(12, 26, 9)).and_then(|m| m.macd_line.last()).copied().unwrap_or(0.0);

            let mut row_cells = vec![
                Cell::from(symbol.as_str()),
                Cell::from(format_price(&price.to_string())),
                Cell::from(Span::styled(
                    format_change(&change.to_string()),
                    Style::default().fg(change_color),
                )),
            ];

            if self.show_indicators {
                row_cells.push(Cell::from(format!("SMA14: {:.2}", sma_value)));
                row_cells.push(Cell::from(Span::styled(
                    format!("RSI14: {:.1}", rsi_value),
                    Style::default().fg(if rsi_value > 70.0 {
                        Color::Red
                    } else if rsi_value < 30.0 {
                        Color::Green
                    } else {
                        Color::Yellow
                    }),
                )));
                row_cells.push(Cell::from(Span::styled(
                    format!("MACD: {:.2}", macd_value),
                    Style::default().fg(if macd_value > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                )));
            }

            Row::new(row_cells)
                .style(if is_selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                })
        });

        let mut headers = vec!["Pair", "Price", "24h Change"];
        if self.show_indicators {
            headers.extend(["SMA(14)", "RSI(14)", "MACD"]);
        }

        let table = Table::new(rows)
            .header(
                Row::new(headers)
                    .style(Style::default().add_modifier(Modifier::BOLD)),
            )
            .block(Block::default().borders(Borders::ALL).title("Live Prices"))
            .widths(&[
                Constraint::Length(10),
                Constraint::Length(15),
                Constraint::Length(12),
            ]);

        f.render_widget(table, area);
    }

    fn render_alerts_view(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let rows = self.alerts.iter().map(|alert| {
            let price = self.prices.get(&alert.symbol)
                .and_then(|p| p.last())
                .map(|p| p.price)
                .unwrap_or(0.0);

            let (condition_text, threshold) = match &alert.condition {
                AlertCondition::PriceAbove(t) => (">", t),
                AlertCondition::PriceBelow(t) => ("<", t),
                AlertCondition::ChangeAbove(t) => ("Δ>", t),
                AlertCondition::ChangeBelow(t) => ("Δ<", t),
                AlertCondition::IndicatorCross { indicator, value, direction: _ } =>
                    (indicator.as_str(), value),
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

    fn render_portfolio_view(&mut self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let total_value = self.portfolio.current_value(&self.prices);
        let profit_loss = self.portfolio.profit_loss(&self.prices);
        let pct_change = (profit_loss / self.portfolio.initial_value) * 100.0;

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
                    Style::default().fg(if profit_loss >= 0.0 { Color::Green } else { Color::Red }),
                ),
            ]),
            Line::from(vec![
                Span::raw("Available Cash: "),
                Span::styled(
                    format!("${:.2}", self.portfolio.cash),
                    Style::default().fg(Color::LightBlue),
                ),
            ]),
        ]))
            .block(Block::default().borders(Borders::ALL).title("Summary"));

        let rows = self.portfolio.holdings.iter().map(|(symbol, qty)| {
            let current_price = self.prices.get(symbol)
                .and_then(|p| p.last())
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
        let selected_period = self.indicator_periods[self.selected_indicator];

        let empty_vec = Vec::new();
        let empty_macd = MacdData {
            macd_line: Vec::new(),
            signal_line: Vec::new(),
            histogram: Vec::new(),
        };

        let sma_values = self.indicators.sma.get(&selected_period).unwrap_or(&empty_vec);
        let rsi_values = self.indicators.rsi.get(&selected_period).unwrap_or(&empty_vec);
        let macd_values = self.indicators.macd.get(&(12, 26, 9)).unwrap_or(&empty_macd);

        let indicator_info = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                format!("Technical Indicators (Period: {})", selected_period),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::raw("SMA: "),
                Span::styled(
                    format!("{:.2}", sma_values.last().unwrap_or(&0.0)),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("RSI: "),
                Span::styled(
                    format!("{:.1}", rsi_values.last().unwrap_or(&0.0)),
                    Style::default().fg(if *rsi_values.last().unwrap_or(&0.0) > 70.0 {
                        Color::Red
                    } else if *rsi_values.last().unwrap_or(&0.0) < 30.0 {
                        Color::Green
                    } else {
                        Color::Magenta
                    }),
                ),
                Span::raw(" (Overbought >70, Oversold <30)"),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("MACD: "),
                Span::styled(
                    format!("{:.2}", macd_values.macd_line.last().unwrap_or(&0.0)),
                    Style::default().fg(if *macd_values.macd_line.last().unwrap_or(&0.0) > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
                Span::raw(" Signal: "),
                Span::styled(
                    format!("{:.2}", macd_values.signal_line.last().unwrap_or(&0.0)),
                    Style::default().fg(Color::Blue),
                ),
            ]),
        ]))
            .block(Block::default().borders(Borders::ALL).title("Indicator Details"));

        f.render_widget(indicator_info, area);
    }

    fn render_footer(&self, f: &mut Frame<CrosstermBackend<io::Stdout>>, area: Rect) {
        let controls = match self.current_view {
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
                Span::styled("t", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Toggle Indicators  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
            DashboardView::Alerts => vec![
                Span::raw("Controls: "),
                Span::styled("v", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Prices  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
            DashboardView::Portfolio => vec![
                Span::raw("Controls: "),
                Span::styled("v", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Prices  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
            DashboardView::Exchanges => vec![
                Span::raw("Controls: "),
                Span::styled("v", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Prices  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Quit"),
            ],
            DashboardView::Indicators => vec![
                Span::raw("Controls: "),
                Span::styled("↑/↓", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" Change Period  "),
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
        |c| format!("{:>7.2}", c),
    )
}