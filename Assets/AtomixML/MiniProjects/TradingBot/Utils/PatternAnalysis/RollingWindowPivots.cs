using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    internal class RollingWindowPivots
    {
        private readonly int _windowSize;
        private readonly Queue<MarketData> _window;

        public RollingWindowPivots(int windowSize)
        {
            _windowSize = windowSize;
            _window = new Queue<MarketData>();
        }

        public void AddDataPoint(MarketData newData)
        {
            _window.Enqueue(newData);
            if (_window.Count > _windowSize)
            {
                _window.Dequeue();
            }

            if (_window.Count == _windowSize)
            {
                CheckForReversal();
            }
        }

        public int CheckForReversal()
        {
            var data = new List<MarketData>(_window);
            int mid = _windowSize / 2;

            if (IsTopReversal(data, mid))
            {
                Console.WriteLine($"🔴 Top detected at {data[mid].Timestamp} | High: {data[mid].High}");

                return 1;
            }

            if (IsBottomReversal(data, mid))
            {
                Console.WriteLine($"🟢 Bottom detected at {data[mid].Timestamp} | Low: {data[mid].Low}");

                return -1;
            }

            return 0;
        }

        private bool IsTopReversal(List<MarketData> data, int mid)
        {
            decimal midHigh = data[mid].High;
            for (int i = 0; i < data.Count; i++)
            {
                if (i != mid && data[i].High >= midHigh)
                    return false;
            }
            return true;
        }

        private bool IsBottomReversal(List<MarketData> data, int mid)
        {
            decimal midLow = data[mid].Low;
            for (int i = 0; i < data.Count; i++)
            {
                if (i != mid && data[i].Low <= midLow)
                    return false;
            }
            return true;
        }
    }
}
