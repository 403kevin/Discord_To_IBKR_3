import schedule
import time
from ib_insync import IB, Contract, Order

# config ===============================
# install schedule package with pip install schedule command
IP_Address = '127.0.0.1'
PORT = 7000
CLIENT_ID = 1
trigger_time = "13:45"  # Set the time in HH:MM format to trigger the function

# config end ===========================

def close_positions():
    # Connect to the TWS or Gateway
    ib = IB()
    ib.connect(IP_Address, PORT, clientId=1)  # Update with your TWS/Gateway connection details

    # Request open positions
    positions = ib.positions()
    if len(positions) == 0:
        print("no positions are available before close all oisition fun call")

    # Process and close positions
    for position in positions:
        if position.position != 0:
            print("********position*********",position)
            print(f"*************Position: {position.contract.symbol}, Quantity: {position.position}, Avg Cost: {position.avgCost}, exchange: {position.contract.exchange}, secType: {position.contract.secType}")
            order = Order()
            order.action = "SELL" if position.position > 0 else "BUY"
            order.orderType = "MKT"  # Market order
            order.totalQuantity = abs(position.position)
            if position.contract.secType == 'OPT':
                print("======== req contract details ===========")
                contract_details = ib.reqContractDetails(position.contract)
                print("contract_details**********",contract_details)
                if contract_details:
                    contract_exchange = contract_details[0].contract.exchange
                    print("Exchange from contract_details:", contract_exchange)
                    print("======== req contract details end ===========")
                    contract = Contract(
                        symbol=position.contract.symbol,
                        secType=position.contract.secType,
                        exchange=contract_exchange,
                        currency=position.contract.currency,
                        lastTradeDateOrContractMonth=position.contract.lastTradeDateOrContractMonth,
                        right=position.contract.right,
                        strike=position.contract.strike
                    )
                else:
                    print("Contract details not available.")
                    continue
            else:
                contract = Contract(symbol=position.contract.symbol, secType=position.contract.secType,
                                exchange="SMART", primaryExchange=position.contract.exchange, currency=position.contract.currency)
            try:
                ib.sleep(1)
                ib.placeOrder(contract, order)
                ib.sleep(1)
                # time.sleep(5)
                print("Order placed successfully.")
            except Exception as e:
                print("Order placement failed:", str(e))
    
    # Disconnect from the TWS or Gateway
    ib.disconnect()


# Schedule the function to be triggered at the specified time
schedule.every().day.at(trigger_time).do(close_positions).tag('close_positions')

# Run the scheduled tasks continuously
while True:
    schedule.run_pending()
    if 'close_positions' in schedule.jobs:
        # The task has been executed, stop the scheduler
        break
    time.sleep(1)
