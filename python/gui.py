import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import relevant modules
from data_reader import load_data
from pair_selectors import get_candidate_pairs, choose_best_pair
from traders import simulate_pair_trading_with_sizing_with_loss_threshold

# --------------------------------------------
# SIMULATION WORKER THREAD CLASS
# --------------------------------------------
class SimulationWorker(QThread):
    updateSignal = pyqtSignal(dict)   # Dictionary of information to be sent on each segment update
    finishedSignal = pyqtSignal(dict) # Results returned when simulation is complete

    def __init__(self, data, initial_lookback, rebalance_interval, loss_threshold,
                 sim_lookback_days, entry_threshold, exit_threshold,
                 cointegration_significance, cointegration_window,
                 position_fraction, starting_capital, method, parent=None):
        super().__init__(parent)
        self.data = data
        self.initial_lookback = initial_lookback
        self.rebalance_interval = rebalance_interval
        self.loss_threshold = loss_threshold
        self.sim_lookback_days = sim_lookback_days
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.cointegration_significance = cointegration_significance
        self.cointegration_window = cointegration_window
        self.position_fraction = position_fraction
        self.starting_capital = starting_capital
        self.method = method
        self._isRunning = True

    def run(self):
        overall_signals = []
        overall_trade_pnls = []
        num_days = len(self.data)
        current_index = self.initial_lookback
        capital = self.starting_capital

        # Simulation loop: A new pair selection will be performed in each segment.
        while current_index < num_days and self._isRunning:
            # New pair selection: Data of size cointegration_window is used for the cointegration test.
            start_idx = max(0, current_index - self.cointegration_window)
            window_data = self.data.iloc[start_idx: current_index]
            candidates = get_candidate_pairs(window_data, significance=self.cointegration_significance)
            current_pair = choose_best_pair(candidates, window_data, self.cointegration_window,
                                            method='ratio', combine_std=True)
            if current_pair is None:
                self.finishedSignal.emit({"error": "No suitable pair found to start the simulation."})
                return
            current_stock1, current_stock2, current_pvalue, current_hedge_ratio = current_pair
            self.updateSignal.emit({
                "message": f"Pair selected: {current_stock1} & {current_stock2}",
                "capital": capital,
                "selected_pair": current_pair,
                "current_index": current_index
            })

            # Segment data: Obtained using sim_lookback_days.
            segment_end_index = min(current_index + self.rebalance_interval, num_days)
            segment_data = self.data.iloc[current_index - self.sim_lookback_days: segment_end_index]
            sim_results = simulate_pair_trading_with_sizing_with_loss_threshold(
                current_stock1, current_stock2, current_hedge_ratio,
                segment_data, lookback_days=self.sim_lookback_days,
                entry_threshold=self.entry_threshold, exit_threshold=self.exit_threshold,
                position_fraction=self.position_fraction, loss_threshold=self.loss_threshold,
                initial_capital=capital, method=self.method
            )
            if sim_results is None:
                self.updateSignal.emit({"message": "Segment simulation error.", "error": True})
                break
            signals, seg_pnl, trade_pnls, updated_capital, trading_days = sim_results
            overall_signals.extend(signals)
            overall_trade_pnls.extend(trade_pnls)
            interval = self.rebalance_interval if trading_days == -1 else trading_days
            segment_end_index = min(current_index + interval, num_days)
            update_info = {
                "segment_start": current_index,
                "segment_end": segment_end_index,
                "seg_pnl": seg_pnl,
                "capital": updated_capital,
                "trade_signals": signals,
                "selected_pair": current_pair,
                "current_index": segment_end_index,
                "message": f"Segment {current_index} to {segment_end_index}: pnl={seg_pnl:.2f}, Capital updated to: {updated_capital:.2f}"
            }
            time.sleep(1)
            self.updateSignal.emit(update_info)
            capital = updated_capital
            current_index = segment_end_index

        percent_return = (((capital / 37.99) / (self.starting_capital / 31.99)) - 1) * 100
        results = {
            'signals': overall_signals,
            'final_capital': capital,
            'percent_return': percent_return,
            'trade_pnls': overall_trade_pnls
        }
        self.finishedSignal.emit(results)

    def stop(self):
        self._isRunning = False

# --------------------------------------------
# PYQT INTERFACE
# --------------------------------------------
class TradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pairs Trading Simulator")
        self.setGeometry(100, 100, 1300, 900)

        self.data = None
        self.data_file = None  # Will store the file path
        self.simWorker = None
        self.current_pair = None
        self.sim_lookback_days = 30
        self.method = None  # Selected method

        # Graph data containers
        self.capital_history = []
        self.time_history = []

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel: Parameters and control buttons
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data_file)
        control_layout.addWidget(self.load_button)

        form = QFormLayout()
        self.starting_capital_edit = QLineEdit("10000")
        self.entry_threshold_edit = QLineEdit("1")
        self.exit_threshold_edit = QLineEdit("0.75")
        self.lookback_days_edit = QLineEdit("252")
        self.rebalance_interval_edit = QLineEdit("15")
        self.cointegration_significance_edit = QLineEdit("0.05")
        self.cointegration_window_edit = QLineEdit("120")
        self.position_fraction_edit = QLineEdit("0.3")
        self.sim_lookback_days_edit = QLineEdit("30")
        self.loss_threshold_edit = QLineEdit("0.6")
        # Add QComboBox for method selection:
        self.method_combo = QComboBox()
        self.method_combo.addItems(["spread", "bollinger_bands", "ratio"])
        form.addRow("Starting Capital:", self.starting_capital_edit)
        form.addRow("Entry Threshold:", self.entry_threshold_edit)
        form.addRow("Exit Threshold:", self.exit_threshold_edit)
        form.addRow("Initial Lookback Days:", self.lookback_days_edit)
        form.addRow("Rebalance Interval:", self.rebalance_interval_edit)
        form.addRow("Cointegration Significance:", self.cointegration_significance_edit)
        form.addRow("Cointegration Window:", self.cointegration_window_edit)
        form.addRow("Position Fraction:", self.position_fraction_edit)
        form.addRow("Sim Lookback Days:", self.sim_lookback_days_edit)
        form.addRow("Loss Threshold:", self.loss_threshold_edit)
        form.addRow("Method:", self.method_combo)
        control_layout.addLayout(form)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)

        self.current_date_label = QLabel("Current Date: -")
        self.selected_pair_label = QLabel("Selected Pair: -")
        self.capital_label = QLabel("Capital: -")
        control_layout.addWidget(self.current_date_label)
        control_layout.addWidget(self.selected_pair_label)
        control_layout.addWidget(self.capital_label)

        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(6)
        self.trade_table.setHorizontalHeaderLabels(["Date", "Operation", "Long Stock", "Short Stock", "Position Size", "P/L"])
        control_layout.addWidget(self.trade_table)

        main_layout.addWidget(control_panel)

        # Right panel: Graphs
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()
        plot_panel.setLayout(plot_layout)

        self.capital_fig, self.capital_ax = plt.subplots()
        self.capital_canvas = FigureCanvas(self.capital_fig)
        plot_layout.addWidget(self.capital_canvas)

        self.price_fig, self.price_ax = plt.subplots()
        self.price_canvas = FigureCanvas(self.price_fig)
        plot_layout.addWidget(self.price_canvas)

        self.spread_fig, self.spread_ax = plt.subplots()
        self.spread_canvas = FigureCanvas(self.spread_fig)
        plot_layout.addWidget(self.spread_canvas)

        main_layout.addWidget(plot_panel)

    def load_data_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "CSV Files (*.csv)")
        if not file_name:
            self.statusBar().showMessage("No file selected.")
            return
        try:
            self.data = load_data(file_name)
            self.data_file = file_name  # Store the file path
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data:\n{e}")
            return
        self.statusBar().showMessage("Data loaded successfully.")

    def start_simulation(self):
        if self.data_file is None:
            QMessageBox.warning(self, "Warning", "Load data first!")
            return

        # Reload the data from file before each simulation start
        try:
            self.data = load_data(self.data_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reloading data:\n{e}")
            return

        try:
            starting_capital = float(self.starting_capital_edit.text())
            initial_lookback = int(self.lookback_days_edit.text())
            rebalance_interval = int(self.rebalance_interval_edit.text())
            loss_threshold = float(self.loss_threshold_edit.text())
            sim_lookback_days = int(self.sim_lookback_days_edit.text())
            entry_threshold = float(self.entry_threshold_edit.text())
            exit_threshold = float(self.exit_threshold_edit.text())
            cointegration_significance = float(self.cointegration_significance_edit.text())
            cointegration_window = int(self.cointegration_window_edit.text())
            position_fraction = float(self.position_fraction_edit.text())
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error in parameters:\n{e}")
            return

        # Determine simulation length as initial lookback + 252
        simulation_length = initial_lookback + 252

        if simulation_length > len(self.data):
            QMessageBox.warning(
                self, 
                "Warning", 
                f"Insufficient rows in data.\nRequired row count: {simulation_length}\nCurrent row count: {len(self.data)}"
            )
            return

        # Get the selected method information (e.g., "spread" or "bollinger_bands")
        self.method = self.method_combo.currentText()

        # Reset log and graph data
        self.trade_table.setRowCount(0)
        self.capital_history = []
        self.time_history = []
        self.capital_ax.clear()
        self.price_ax.clear()
        self.spread_ax.clear()
        self.capital_canvas.draw()
        self.price_canvas.draw()
        self.spread_canvas.draw()
        self.selected_pair_label.setText("Selected Pair: -")
        self.current_date_label.setText("Current Date: -")
        self.capital_label.setText(f"Capital: {starting_capital:.2f}")

        # Use only the most recent 'simulation_length' rows
        self.data = self.data.tail(simulation_length)

        self.sim_lookback_days = sim_lookback_days
        self.capital_history = [starting_capital]
        self.time_history = [self.data.index[initial_lookback - 1]]

        self.simWorker = SimulationWorker(
            data=self.data,
            initial_lookback=initial_lookback,
            rebalance_interval=rebalance_interval,
            loss_threshold=loss_threshold,
            sim_lookback_days=sim_lookback_days,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            cointegration_significance=cointegration_significance,
            cointegration_window=cointegration_window,
            position_fraction=position_fraction,
            starting_capital=starting_capital,
            method=self.method
        )
        self.simWorker.updateSignal.connect(self.handle_update)
        self.simWorker.finishedSignal.connect(self.handle_finished)
        self.simWorker.start()
        self.statusBar().showMessage("Simulation started.")

    def stop_simulation(self):
        if self.simWorker is not None:
            self.simWorker.stop()
            self.simWorker.wait()
            self.statusBar().showMessage("Simulation stopped.")

    def handle_update(self, info):
        if "error" in info:
            self.statusBar().showMessage(info.get("message", "Unknown error"))
            return

        if "current_index" in info:
            idx = info["current_index"]
            if idx < len(self.data):
                current_date = self.data.index[idx]
                self.current_date_label.setText(f"Current Date: {current_date.strftime('%Y-%m-%d')}")
        if "capital" in info:
            cap = info["capital"]
            self.capital_label.setText(f"Capital: {cap:.2f}")
            if "segment_end" in info:
                seg_end = info["segment_end"]
                date = self.data.index[seg_end - 1]
                self.time_history.append(date)
                self.capital_history.append(cap)
                self.capital_ax.clear()
                self.capital_ax.plot(self.time_history, self.capital_history, label="Capital")
                self.capital_ax.set_title("Capital Graph")
                self.capital_ax.set_xlabel("Date")
                self.capital_ax.set_ylabel("Capital")
                self.capital_ax.legend()
                self.capital_canvas.draw()

        if "selected_pair" in info and info["selected_pair"]:
            pair = info["selected_pair"]
            self.current_pair = pair
            self.selected_pair_label.setText(f"Selected Pair: {pair[0]} - {pair[1]}")
        if "message" in info:
            self.statusBar().showMessage(info["message"])

        # When adding trade log rows, determine which stock is long and which is short based on the operation.
        if "trade_signals" in info:
            for sig in info["trade_signals"]:
                row = self.trade_table.rowCount()
                self.trade_table.insertRow(row)
                # Sig tuple example: (timestamp, operation, zscore, position, price, [trade_return])
                timestamp = sig[0]
                operation = sig[1]
                if self.current_pair:
                    if operation == "Enter Long":
                        long_stock = self.current_pair[0]
                        short_stock = self.current_pair[1]
                    elif operation == "Enter Short":
                        long_stock = self.current_pair[1]
                        short_stock = self.current_pair[0]
                    else:
                        long_stock = self.current_pair[0]
                        short_stock = self.current_pair[1]
                else:
                    long_stock = ""
                    short_stock = ""
                position = sig[3] if len(sig) > 3 else ""
                trade_return = sig[5] if len(sig) > 5 else ""
                self.trade_table.setItem(row, 0, QTableWidgetItem(str(timestamp)))
                self.trade_table.setItem(row, 1, QTableWidgetItem(str(operation)))
                self.trade_table.setItem(row, 2, QTableWidgetItem(str(long_stock)))
                self.trade_table.setItem(row, 3, QTableWidgetItem(str(short_stock)))
                self.trade_table.setItem(row, 4, QTableWidgetItem(str(position)))
                self.trade_table.setItem(row, 5, QTableWidgetItem(str(trade_return)))

        if self.current_pair and "current_index" in info:
            current_idx = info["current_index"]
            window_start = max(0, current_idx - self.sim_lookback_days)
            price_window = self.data.iloc[window_start: current_idx]
            stock1, stock2, _, hedge_ratio = self.current_pair

            self.price_ax.clear()
            self.price_ax.plot(price_window.index, price_window[stock1], label=stock1)
            self.price_ax.plot(price_window.index, price_window[stock2], label=stock2)
            self.price_ax.set_title("Stock Prices")
            self.price_ax.set_xlabel("Date")
            self.price_ax.set_ylabel("Price")
            self.price_ax.legend()
            self.price_canvas.draw()

            # Draw the spread graph based on the method parameter:
            self.spread_ax.clear()
            if self.method == "spread" or self.method == "bollinger_bands":
                # Only the spread graph
                spread = price_window[stock1] - hedge_ratio * price_window[stock2]
                self.spread_ax.plot(price_window.index, spread, label="Spread")
                self.spread_ax.set_title("Spread Graph")
                self.spread_ax.set_xlabel("Date")
                self.spread_ax.set_ylabel("Spread")
            self.spread_ax.legend()
            self.spread_canvas.draw()


    def handle_finished(self, results):
        if "error" in results:
            QMessageBox.critical(self, "Simulation Error", results["error"])
        else:
            # TL cinsinden final capital'ı dolar cinsine çevirmek için 37,99'a bölüyoruz
            final_capital_usd = results['final_capital'] / 37.99
            msg = (f"Final Capital (TL): {results['final_capital']:.2f}\n"
                f"Final Capital (USD): {final_capital_usd:.2f}\n"
                f"Return: {results['percent_return']:.2f}%\n"
                f"Trades: {len(results['trade_pnls'])}\n\n"
                "Do you want to save the trade logs as CSV?")
            reply = QMessageBox.question(self, "Simulation Completed", msg,
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.save_trade_log_to_csv()
        self.statusBar().showMessage("Simulation completed.")


    def save_trade_log_to_csv(self):
        # Read the data from the trade log table
        rows = self.trade_table.rowCount()
        cols = self.trade_table.columnCount()
        headers = [self.trade_table.horizontalHeaderItem(i).text() for i in range(cols)]
        data = []
        for row in range(rows):
            row_data = {}
            for col in range(cols):
                item = self.trade_table.item(row, col)
                row_data[headers[col]] = item.text() if item is not None else ""
            data.append(row_data)
        df = pd.DataFrame(data)
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Trade Logs as CSV", "", "CSV Files (*.csv)")
        if file_name:
            try:
                df.to_csv(file_name, index=False)
                QMessageBox.information(self, "Success", "Trade logs saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error occurred while saving trade logs: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TradingGUI()
    window.show()
    sys.exit(app.exec_())
