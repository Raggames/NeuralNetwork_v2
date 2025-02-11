using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot.Data.TwelveDataAPI
{
    public class TwelveDataAPIStreamer 
    {
        private CancellationTokenSource _tokenSource;
        private Action<MarketData> _onOHLCUpdateCallback;

        private const string _url = "https://api.twelvedata.com/time_series?symbol=&interval=1min&start_date=2025-02-10&outputsize=5000&apikey=96b90e1bd0d141a089a9660d19a92da2";

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
