import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import itertools


class vanilla_bond:
    def __init__(self,cpr,**kwargs):
        """
        ytm: annualized yield to maturity
        term: maturity in year
        cpr: annualized coupon rate
        frequency: coupon payment frequency
        dcf_dict: a dictionary in the form of {time: discount factor} or {time: rate}, here, time are coupon payment days
        mode: either 'discount factor' or 'rate', depending on the values in dcf_dict
        """
        self.cpr = cpr

        if (set(kwargs) == {'dcf_dict','mode'}): 
            self.definition = 'dcf'
            self.dcf_dict = kwargs['dcf_dict']
            self.mode = kwargs['mode']
            if self.mode not in ['rate','discount factor']:
                print("if you use 'dcf_dict' to define a bond, mode must be either 'rate' or 'discount factor'.")
                raise ValueError

            cp_date = sorted(self.dcf_dict)
            time_diff = [cp_date[ind] - cp_date[ind-1] for ind in range(1,len(cp_date))]
            if (len(set(time_diff)) != 1) or (time_diff[0] != cp_date[0]):
                print('not equally spaced payment dates')
                raise ValueError
            self.freq = cp_date[0]
            self.term = max(cp_date)

        elif (set(kwargs) == {'ytm','term','freq'}):
            self.definition = 'normal'
            self.ytm = kwargs['ytm']
            self.term = kwargs['term']
            self.freq = kwargs['freq']
        else:
            print("the keys of your input dictionary should be either ['ytm','term','freq'] or ['dcf_dict','mode']")
            raise ValueError
        assert self.term%self.freq == 0
    

    def get_ytm(self,threshold=0.0001):
        if self.definition == 'normal':
            return self.ytm
        else:
            def test_price(ytm):
                pri = 1/(1+ytm*self.freq)**(self.term/self.freq)
                for i in range(1,int(self.term/self.freq)+1):
                    pri += self.cpr*self.freq/(1+ytm*self.freq)**i
                return pri
            low = 0
            high = 1
            price = self.get_price()
            
            while abs(test_price(low) - price) > threshold:
                low_d = test_price(low) - price
                high_d = test_price(high) - price
                assert low_d * high_d < 0
                new = (low + high)/2
                new_d = test_price(new) - price
                if new_d * low_d > 0:
                    low = new
                else:
                    high = new
            return low

    def get_price(self):
        if self.definition == 'dcf':
            if self.mode == 'discount factor':
                price = self.dcf_dict[self.term]
                for i in self.dcf_dict:
                    price += self.dcf_dict[i]*self.cpr*self.freq
                return price
            elif self.mode == 'rate':
                price = 1/(1+self.dcf_dict[self.term]*self.freq)**(self.term/self.freq)
                for i in self.dcf_dict:
                    price += self.cpr*self.freq/(1+self.dcf_dict[i]*self.freq)**(i/self.freq)
                return price
            else:
                print('mode unavailable')
                return
        elif self.definition == 'normal':
                
                price = 1/(1+self.ytm*self.freq)**(self.term/self.freq)
                for i in range(1,int(self.term/self.freq)+1):
                    price += self.cpr*self.freq/(1+self.ytm*self.freq)**i
                return price
    
    def get_macd(self):
        if self.definition == 'dcf':
            if self.mode == 'discount factor':
                pvcf_t = self.dcf_dict[self.term] * self.term
                for i in self.dcf_dict:
                    pvcf_t += self.dcf_dict[i] * self.cpr * self.freq * i 
            elif self.mode == 'rate':                
                pvcf_t = (1/(1+self.dcf_dict[self.term]*self.freq)**(self.term/self.freq)) * self.term
                for i in self.dcf_dict:
                    pvcf_t += (self.cpr*self.freq/(1+self.dcf_dict[i]*self.freq)**(i/self.freq)) * i
            else:
                print('mode unavailable')
                return
        elif self.definition == 'normal':
                pvcf_t = (1/(1+self.ytm*self.freq)**(self.term/self.freq)) * self.term
                for i in range(1,int(self.term/self.freq)+1):
                    pvcf_t += (self.cpr*self.freq/(1+self.ytm*self.freq)**i) * (i*self.freq)
        return pvcf_t/self.get_price()
    
    def get_modd(self):
        return self.get_macd()/(1+self.get_ytm()*self.freq)
    
    def get_conv(self,delta_ytm=0.0001):
        V_minus = vanilla_bond(self.cpr,ytm=self.get_ytm()-delta_ytm,term=self.term,freq=self.freq).get_price()
        V_plus = vanilla_bond(self.cpr,ytm=self.get_ytm()+delta_ytm,term=self.term,freq=self.freq).get_price()
        V = self.get_price()
        return (V_plus+ V_minus - 2*V)/(V*(delta_ytm**2))
    
    def first_order_estimate(self,d_ytm=0.01/100):
        ytm = self.get_ytm()
        dp_pct = - self.get_modd() * d_ytm
        return dp_pct 
    
    def second_order_estimate(self,d_ytm=0.01/100,delta_ytm=0.0001):
        ytm = self.get_ytm()
        dp_pct = - self.get_modd() * d_ytm + 0.5 * self.get_conv(delta_ytm) * ((d_ytm)**2)
        return dp_pct


    def get_par_rate(self,threshold=0.0001):
        def test_price(ytm):
            pri = 1/(1+ytm*self.freq)**(self.term/self.freq)
            for i in range(1,int(self.term/self.freq)+1):
                pri += self.cpr*self.freq/(1+ytm*self.freq)**i
            return pri
        low = 0
        high = 1
        price = 1
        
        while abs(test_price(low) - price) > threshold:
            low_d = test_price(low) - price
            high_d = test_price(high) - price
            assert low_d * high_d < 0
            new = (low + high)/2
            new_d = test_price(new) - price
            if new_d * low_d > 0:
                low = new
            else:
                high = new
        return low
    
    def get_interest_risk(self,max_decr=None,max_incr=None,interval=None):
        if max_decr is None:
            max_decr = self.get_ytm()
        if max_incr is None:
            max_incr = self.get_ytm() * 3
        if interval is None:
            interval = max(int((max_incr - max_decr)/0.0001),100)
        ytm_grid = np.linspace(-max_decr,max_incr,interval)
        ytm = self.get_ytm()
        df_int_risk = pd.DataFrame(ytm_grid,columns=['d_ytm'])
        df_int_risk['1st_estimate'] = df_int_risk['d_ytm'].apply(lambda d_ytm: self.first_order_estimate(d_ytm=d_ytm))
        df_int_risk['2nd_estimate'] = df_int_risk['d_ytm'].apply(lambda d_ytm: self.second_order_estimate(d_ytm=d_ytm))
        df_int_risk['real_price'] = df_int_risk['d_ytm'].apply(lambda d_ytm: vanilla_bond(cpr=self.cpr,
                                                                                          ytm=ytm+d_ytm,
                                                                                          term=self.term,
                                                                                          freq=self.freq).get_price()/self.get_price()-1)
        trace_first = go.Scatter(x=df_int_risk['d_ytm']+ytm,y=df_int_risk['1st_estimate'],mode='lines',name='1st_estimate_pct_change')
        trace_second = go.Scatter(x=df_int_risk['d_ytm']+ytm,y=df_int_risk['2nd_estimate'],mode='lines',name='2nd_estimate_pct_change')
        trace_real = go.Scatter(x=df_int_risk['d_ytm']+ytm,y=df_int_risk['real_price'],mode='lines',name='real_price')
        data = [trace_first,trace_second,trace_real]
        fig = go.Figure(data=data)
        fig.update_layout(xaxis_title='yield to maturity',
                          yaxis_title='price pct_change',
                          title=f'interest risk study ({self.term} years, {self.cpr} coupon, {self.freq} years componding frequency at ytm {self.get_ytm()})')
        fig.show()