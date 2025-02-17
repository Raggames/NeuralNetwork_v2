using Newtonsoft.Json;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{

    [Serializable]
    public class MarketDatas
    {
        public List<MarketData> Datas;
    }

    [Serializable]
    public class MarketData
    {
        /// <summary>
        /// The exact date and time of the price data point. In intraday trading, this is in 1-minute intervals (e.g., "2024-02-07 09:30:00").
        /// </summary>
        [JsonProperty("Timestamp")]
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// The price at which the stock/crypto opened for that time period (minute, hour, day).
        /// </summary>
        [JsonProperty("Open")]
        public decimal Open { get; set; }

        /// <summary>
        /// The highest price reached during that time period.
        /// </summary>
        [JsonProperty("High")]
        public decimal High { get; set; }

        /// <summary>
        /// The lowest price reached during that time period.
        /// </summary>
        [JsonProperty("Low")]
        public decimal Low { get; set; }

        /// <summary>
        /// The price at which the stock/crypto closed at the end of that time period.
        /// </summary>
        [JsonProperty("Close")]
        public decimal Close { get; set; }

        /// <summary>
        // The number of shares/units traded in that time period. Higher volume means more liquidity and less price manipulation risk.
        /// </summary>
        [JsonProperty("Volume")]
        public int Volume { get; set; }


        [JsonIgnore] public bool isBullish => Open < Close;
        [JsonIgnore] public bool isBearish => Open > Close;
                    
    }

}
