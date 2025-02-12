using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public struct TransactionData
    {
        public string AssetSymbol { get; set; }  // e.g., "AAPL", "BTCUSD"
        public decimal Price { get; set; }       // Transaction price
        public decimal Volume { get; set; }      // Number of shares/contracts/coins
        public DateTime Timestamp { get; set; }  // Time of transaction
        public BuySignals Signal { get; set; }

        public TransactionData(string assetSymbol, decimal price, decimal volume, DateTime timestamp, BuySignals signal)
        {
            AssetSymbol = assetSymbol;
            Price = price;
            Volume = volume;
            Timestamp = timestamp;
            Signal = signal;
        }

        public override string ToString()
        {
            return $"{Timestamp}: {Signal} {Volume} of {AssetSymbol} at {Price:C}";
        }
    }
}
