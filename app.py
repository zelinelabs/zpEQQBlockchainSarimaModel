from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import hashlib
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Supply chain optimization functions
def calculate_stochastic_eoq(demand_mean, demand_std, setup_cost, holding_cost, service_level, lead_time):
    z = -np.log(1 - service_level)
    safety_stock = z * demand_std * np.sqrt(lead_time)
    eoq = np.sqrt((2 * demand_mean * setup_cost) / holding_cost)
    return eoq + safety_stock

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash, nonce):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash
        self.nonce = nonce

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = self.mine_block(0, "0", time.time(), "Genesis Block")
        self.chain.append(genesis_block)

    def hash_block(self, index, previous_hash, timestamp, data, nonce):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data) + str(nonce)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def mine_block(self, index, previous_hash, timestamp, data):
        nonce = 0
        while True:
            hash = self.hash_block(index, previous_hash, timestamp, data, nonce)
            if hash[:self.difficulty] == '0' * self.difficulty:
                return Block(index, previous_hash, timestamp, data, hash, nonce)
            nonce += 1

    def add_block(self, data):
        previous_block = self.chain[-1]
        index = len(self.chain)
        timestamp = time.time()
        previous_hash = previous_block.hash
        block = self.mine_block(index, previous_hash, timestamp, data)
        self.chain.append(block)

def train_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(data['Demand'], order=order, seasonal_order=seasonal_order)
    sarima_model = model.fit(disp=False)
    return sarima_model

def forecast_demand(model, periods):
    forecast = model.forecast(steps=periods)
    return forecast

def simulate_supply_chain(demand_forecast, eoq, setup_cost, holding_cost, lead_time):
    inventory_level = 0
    total_cost = 0
    orders = []
    
    for period, demand in enumerate(demand_forecast):
        if inventory_level < demand:
            order_quantity = eoq
            orders.append((period, order_quantity))
            inventory_level += order_quantity
            total_cost += setup_cost
        inventory_level -= demand
        total_cost += holding_cost * inventory_level

    return orders, total_cost

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    annual_demand_mean = float(request.form['annual_demand_mean'])
    demand_std = float(request.form['demand_std'])
    setup_cost = float(request.form['setup_cost'])
    holding_cost = float(request.form['holding_cost'])
    service_level = float(request.form['service_level'])
    lead_time = int(request.form['lead_time'])

    # Part 1: Stochastic EOQ Calculation
    eoq = calculate_stochastic_eoq(annual_demand_mean, demand_std, setup_cost, holding_cost, service_level, lead_time)
    
    # Part 2: Blockchain
    blockchain = Blockchain()
    blockchain.add_block({"event": "Order Placed", "quantity": eoq})
    blockchain.add_block({"event": "Order Received", "quantity": eoq})
    
    blockchain_data = [{"index": block.index, "hash": block.hash, "nonce": block.nonce} for block in blockchain.chain]

    # Part 3: Advanced Machine Learning Demand Forecasting
    data = pd.DataFrame({
        'Period': np.arange(1, 61),
        'Demand': np.random.poisson(lam=200, size=60)
    })
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    model = train_sarima_model(train_data)
    forecasted_demand = forecast_demand(model, len(test_data))
    
    # Part 4: Simulation for Performance Evaluation
    orders, total_cost = simulate_supply_chain(forecasted_demand, eoq, setup_cost, holding_cost, lead_time)

    # Plotting the forecasted demand and orders
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(forecasted_demand)), forecasted_demand, label='Forecasted Demand')
    plt.scatter(*zip(*orders), color='red', label='Orders', zorder=5)
    plt.xlabel('Period')
    plt.ylabel('Units')
    plt.legend()
    plt.title('Forecasted Demand and Orders')
    
    # Save plot to BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', eoq=eoq, blockchain_data=blockchain_data, total_cost=total_cost, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
