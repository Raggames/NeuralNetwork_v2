using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.TradingBot.Data.TwelveDataAPI
{
    public class TwelveDataAPIStreamer 
    {
        private CancellationTokenSource _tokenSource;
        private Action<MarketData> _onOHLCUpdateCallback;

        private const string _url = "";

        public async Task<StockDataResponse> GetHistoricalData(string symbol, string interval, string startDate = "2025-02-10")
        {
            string url = $"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&start_date={startDate}&outputsize=5000&apikey=96b90e1bd0d141a089a9660d19a92da2";

            using (HttpClient client = new HttpClient())
            {
                try
                {
                    HttpResponseMessage response = await client.GetAsync(url);
                    response.EnsureSuccessStatusCode();
                    string responseData = await response.Content.ReadAsStringAsync();
                    return JsonConvert.DeserializeObject<StockDataResponse>(responseData);  
                }
                catch (HttpRequestException e)
                {
                    Debug.LogError($"Request error: {e.Message}");

                    return null;
                }
            }
        }

        public async void StartExecution(Action<MarketData> onOHLCUpdateCallback)
        {
            if (_tokenSource != null) StopExecution();


            _onOHLCUpdateCallback = onOHLCUpdateCallback;

            _tokenSource = new CancellationTokenSource();
            await Update(_tokenSource.Token);    
        }

        public void StopExecution()
        {
            _tokenSource?.Cancel(); 
        }

        async Task Update(CancellationToken cancellationToken)
        {
            while (true)
            {
                //await 


                await Task.Delay(5);

                if(cancellationToken.IsCancellationRequested) return;
            }

        }


    }
}
