using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class TransactionData
    {
        public string AssetSymbol { get; set; }  // e.g., "AAPL", "BTCUSD"
        public decimal EntryPrice { get; set; }       // Transaction price
        public decimal ExitPrice { get; set; }       // Transaction price
        public decimal Volume { get; set; }      // Number of shares/contracts/coins
        public DateTime Timestamp { get; set; }  // Time of transaction
        public PositionTypes Signal { get; set; }


        public decimal Balance => (ExitPrice - EntryPrice) * Volume;

        public TransactionData(string assetSymbol, decimal entryPrice, decimal volume, DateTime timestamp, PositionTypes signal)
        {
            AssetSymbol = assetSymbol;
            EntryPrice = entryPrice;
            Volume = volume;
            Timestamp = timestamp;
            Signal = signal;
        }

        public override string ToString()
        {
            return $"{Timestamp}: {Signal} {Volume} of {AssetSymbol}. Entry {EntryPrice}, Exit {ExitPrice}. Balance {Balance}";
        }
    }
}
