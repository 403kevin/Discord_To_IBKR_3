# =============================================================================================== #
import logging
from datetime import datetime,timedelta
import utils
import polygon
import re
import config
from ib_insync import  Contract, Order

# =============================================================================================== #

def close_positions(self):
        # Request open positions
        positions = self.ib_interface.get_positions()
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
                    contract_details = self.ib_interface.get_ContractDetails(position.contract)
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
                    self.ib_interface.sleepForMinute()
                    self.ib_interface.place_order_to_close_all(contract, order)
                    self.ib_interface.sleepForMinute()
                    # time.sleep(5)
                    print("Order placed successfully to close position.")
                except Exception as e:
                    print("Order placement failed to close position:", str(e))

class CommonParser:
    def __init__(self):
       pass

    @staticmethod
    def parse_message(self,msg: dict, state: dict = None) -> dict:
        print("************** msg common parser 23 FEB *******",msg)
        print("**********state********",state)
        msgg = ""
        try:
            msgg += msg['content']
        except:
            pass
        try:
            msgg += " " + msg['embeds'][0]["title"]
        except:
            pass
        try:
            msgg += " " + msg['embeds'][0]["description"]
        except:
            pass
        for word in self.REJECT_SIGNALS:
            if word.lower() in msgg.lower():
                print("skipping signal due to reject word")
                logging.warning("skipping signal due to reject word")
                return
        # if "all out" in msgg.lower():
        #     close_positions(self)
        #     return
        print("msgg>>>>>>>>>>>",msgg)
        contains_signal = any(substring in msgg.lower() for substring in config.SELL_SIGNALS) or any(substring in msgg.lower() for substring in config.TRIM_SIGNALS)
        if  contains_signal and config.ONE_CONTRACT_AT_A_TIME:
            print("inside close position<<<<<<<<<<<<<<<<<<<")
            logging.warning("close all position called>>>>>>>>>>>>")
            close_positions(self)
            return {}
        if ("ticker" in msgg or "Ticker" in msgg or "TICKER" in msgg) and ("expiration" in msgg or "Expiration" in msgg or "EXPIRATION" in msgg) :
            # if "expiration" in msgg or "Expiration" in msgg or "EXPIRATION" in msgg:
                split_msg = re.split(r'[\s\n:*]+', msgg)
                split_msg = [item for item in split_msg if item != ""]
                split_msg = [item for item in split_msg if item != "N/A"]
                split_msg = [item for item in split_msg if item != "@"]
                print("split_msg",split_msg)
                j=0
                for ele in split_msg:
                    if "ticker" in ele or "Ticker" in ele or "TICKER" in ele :
                        ticker = split_msg[j+1] 
                    j += 1
                
                for sig in split_msg:
                    if "/" in sig:
                        exp_month, exp_day = sig.split("/")
                        # split_msg.remove(sig)
                        index_of_slash_element = sig.strip()
                        break
                for msg in split_msg:
                    if re.match(r'\d+(\.\d+)?[a-zA-Z]$', msg.strip()):
                        p_or_c = msg[-1].lower()
                        strike = round(float(msg[:-1]), 2)
                for element in split_msg:
                    if element.lower() in self.BUY_SIGNALS:
                        if split_msg.index(element) < split_msg.index(index_of_slash_element):
                            instr = 'BUY'
                            split_msg.remove(element)
                            break
                    if element.lower() in self.SELL_SIGNALS:
                        if split_msg.index(element) < split_msg.index(index_of_slash_element):
                            instr = 'SELL'
                            split_msg.remove(element)
                            break
                    if element.lower() in self.TRIM_SIGNALS:
                        if split_msg.index(element) < split_msg.index(index_of_slash_element):
                            instr = 'TRIM'
                            split_msg.remove(element)
                            break
                return {'underlying': ticker, 'exp_month': int(exp_month), 'exp_day': int(exp_day),
                    'strike': strike, 'p_or_c': p_or_c, 'instr': instr, 'id': state['msg_id']}

        msg_split = re.split(r'[\s\n:*]+|\*\*', msgg)
        msg_split = ["" if value == "exp" else value for value in msg_split]
        msg_split = ["" if value == "N/A" else value for value in msg_split]
        msg_split = ["" if value == "@" else value for value in msg_split]
        msg_split = [item for item in msg_split if item != '']

        # ====== Add this block ======
        # Handle DTE (Days to Expiry) format (e.g., "0DTE", "1DTE")
        for i in range(len(msg_split)):
            if "DTE" in msg_split[i]:
                try:
                    dte_days = int(msg_split[i].replace("DTE", "").strip())
                    expiry_date = utils.get_business_day(dte_days)  # From utils.py
                    # Replace "0DTE" with "MM/DD" format
                    msg_split[i] = f"{expiry_date.month}/{expiry_date.day}"
                except ValueError:
                    pass
        # ====== End of added block ======

        print("msg_split2",msg_split)
        instr = ""
        exp_month = ""
        exp_day = ""
        ticker = ""
        strike = ""
        p_or_c = ""
        order_limit = 0
        
        k = 0
        for msg in msg_split:
            if "DTE" in msg:
                days = msg.split("DTE")[0]
                today = datetime.today()
                current_dt = today + timedelta(days=int(days))
                print("current_dt",current_dt)
                exp_month, exp_day = current_dt.month, current_dt.day
                print("exp_month",exp_month,"exp_day",exp_day)
                exp_date = f"{exp_month}/{exp_day}"
                msg_split = [exp_date if value == msg else value for value in msg_split]

        # if "0DTE" in msg_split:
        #     current_dt = datetime.today()
        #     exp_month, exp_day = current_dt.month, current_dt.day
        #     exp_date = f"{exp_month}/{exp_day}"
        #     msg_split = [exp_date if value == "0DTE" else value for value in msg_split]
        print("msg_split2----------1----",msg_split)
        for msg in msg_split:
            match = re.match(r'(\d+(\.\d+)?)call$', msg.strip()) or re.match(r'(\d+(\.\d+)?)put$', msg.strip())
    
            if match:
                strike = float(match.group(1))
                p_or_c = "c" if "call" in msg else "p"
                ticker = msg_split[k-2]
                order_limit = k
                break
            k += 1
        print("msg_split2----------2----",ticker)

        if p_or_c == "":
            k = 0 
            for msg in msg_split:
                if re.match(r'\d+(\.\d+)?[a-zA-Z]$', msg.strip()):
                    p_or_c = msg[-1].lower()
                    strike = round(float(msg[:-1]), 2)
                    order_limit = k
                    break
                k += 1
        t = 0
        print("msg_split2----------3----",ticker)

        if p_or_c == "":
            for msg in msg_split:
                if re.match(r'\$\d+(\.\d+)?[a-zA-Z]', msg.strip()):
                    p_or_c = msg[-1].lower()
                    strike = round(float(msg[1:-1]), 2)
                    order_limit = t
                    msg_split.remove(msg)
                    msg_split.insert(t, msg[1:])
                    break
                t += 1
        print("msg_split************",ticker)

        print("msg_split2----------4----",ticker)

        f = 0
        if strike == "":
            for msg in msg_split:
                if re.match(r'^\$\d+(\.\d+)?$', msg.strip()):
                    strike = round(float(msg[1:]), 2)
                    order_limit = f
                    msg_split.remove(msg)
                    msg_split.insert(f, msg[1:])
                    if re.match(r'^[a-zA-Z]+$',msg_split[f-1]):
                        ticker = msg_split[f-1]
                    break
                f += 1

        print("p_or_c___________",p_or_c,"strike____________",strike)
        print("msg_split2----------5----",ticker)
        e = 0 
        if p_or_c == "":
            for msg in msg_split:
                if msg.lower() == "c" or msg.lower() == "call" or msg.lower() == "calls" :
                    p_or_c = "c"
                    if strike == "":
                        strike = round(float(msg_split[e-1]), 2)
                    order_limit = e
                    if re.match(r'^[a-zA-Z]+$',msg_split[e-2]):
                        ticker = msg_split[e-2]
                    break
                if msg.lower() == "p" or msg.lower() == "put" or msg.lower() == "puts" :
                    p_or_c = "p"
                    if strike == "":
                        strike = round(float(msg_split[e-1]), 2)
                    order_limit = e
                    if re.match(r'^[a-zA-Z]+$',msg_split[e-2]):
                        ticker = msg_split[e-2]
                    break
                e += 1
        l = 0
        print("msg_split2----------6----",ticker)

        for element in msg_split:
            if element.lower() in self.BUY_SIGNALS and l < order_limit:
                instr = 'BUY'
                msg_split.remove(element)
                break
            if element.lower() in self.SELL_SIGNALS and l < order_limit:
                instr = 'SELL'
                msg_split.remove(element)
                break
            if element.lower() in self.TRIM_SIGNALS and l < order_limit:
                instr = 'TRIM'
                msg_split.remove(element)
                break
            l += 1
        # msg_split.pop(0)
        print("msg_split2----------7----",ticker)
        if ticker == "":
            for elem in msg_split:
                if re.match(r'^\$[a-zA-Z]+$', elem.strip()):
                    ticker = elem.strip().upper()[1:]
                    break
        print("msg_split2----------8----",ticker)

        if ticker == "":
            i=0
            for item in msg_split:
                # print("i******",i,item)
                if re.match(r'\d+(\.\d+)?[a-zA-Z]$', item.strip()):
                    if i == 0 :
                        if re.match(r'^[a-zA-Z]+$',msg_split[0] ) and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                            ticker = msg_split[i-1]
                            # print("ticker*******in upper 111*****",ticker)

                    elif re.match(r'^[a-zA-Z]+$',msg_split[i-1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                        ticker = msg_split[i-1]
                        # print("ticker*******in upper22222*****",ticker)
                if "/" in item:
                    if i == 0 :
                        if re.match(r'^[a-zA-Z]+$',msg_split[0] ) and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                            ticker = msg_split[i-1]
                            # print("ticker*******in lower1111*****",ticker)
                    elif re.match(r'^[a-zA-Z]+$',msg_split[i-1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                        ticker = msg_split[i-1]
                        # print("ticker*******in lower2222*****",ticker)
                        break
                i += 1
            print("msg_split2----------9----",ticker)

            if len(ticker) == 0:
                i=0
                for item in msg_split:
                    # print("i******",i,item)
                    if re.match(r'\d+(\.\d+)?[a-zA-Z]$', item.strip()):
                        if i == 0 :
                            if re.match(r'^[a-zA-Z]+$',msg_split[i+1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                                ticker = msg_split[i+1]
                        if i != 0 :
                            if re.match(r'^[a-zA-Z]+$',msg_split[i+1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                                ticker = msg_split[i+1]
                    if "/" in item:
                        if i != 0 :
                            if re.match(r'^[a-zA-Z]+$',msg_split[i+1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                                ticker = msg_split[i+1]
                    i += 1
            print("msg_split2----------10----",ticker)
            if len(ticker) == 0:
                i=0
                for item in msg_split:
                    if item.lower() in self.BUY_SIGNALS or item.lower() in self.SELL_SIGNALS or item.lower() in self.TRIM_SIGNALS :

                        if re.match(r'^[a-zA-Z]+$',msg_split[i-1] )and item.strip() not in self.BUY_SIGNALS and item.strip() not in self.SELL_SIGNALS and item.strip() not in self.TRIM_SIGNALS:
                                ticker = msg_split[i-1]
                    i += 1

        if ticker.lower() not in config.DAILY_EXPIRY_SIGNALS :
            date =  utils.get_next_friday(symbol='')
            exp_month = int(date[4:6])
            exp_day = int(date[6:8]) 
        elif ticker.lower() in config.DAILY_EXPIRY_SIGNALS :
            current_dt = datetime.today()
            exp_month, exp_day = current_dt.month, current_dt.day
        for sig in msg_split:
            if "/" in sig:
                splitted_date = sig.split("/")
                exp_month, exp_day = splitted_date[0],splitted_date[1]
                # msg_split.remove(sig)
                break
        print("msg_split2----------11----",ticker)

        # if exp_month == "":
        #     # current_dt = datetime.today()
        #     # exp_month, exp_day = current_dt.month, current_dt.day
        #     date =  utils.get_next_friday(symbol='')
        #     exp_month = int(date[4:6])
        #     exp_day = int(date[6:8]) 
        if instr == "" :
            if self.FORMAT_12_BUY:
                instr = 'BUY'
        print("exp_month=====",exp_month,type)
        final_obj = {'underlying': ticker, 'exp_month': int(exp_month), 'exp_day': int(exp_day),
                'strike': strike, 'p_or_c': p_or_c, 'instr': instr, 'id': state['msg_id']}   

        print("============ final object common parser==========",final_obj)
        if instr == "" and msg_split and msg_split[0].upper() in config.BUY_SIGNALS:
            instr = "BUY"
        elif instr == "":
            print("==========================")
            print(" order instruction not found")
            print("==========================")

        return {'underlying': ticker, 'exp_month': int(exp_month), 'exp_day': int(exp_day),
                'strike': strike, 'p_or_c': p_or_c, 'instr': instr, 'id': state['msg_id']}

    
# =============================================================================================== #

# MONTHS_ABBR = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7,
#                'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12, 'sept': 9}
# PUT_CALL_ABBR = {'put': 'P', 'call': 'C', 'puts': 'P', 'calls': 'C'}


# =============================================================================================== #


if __name__ == '__main__':
    pass
