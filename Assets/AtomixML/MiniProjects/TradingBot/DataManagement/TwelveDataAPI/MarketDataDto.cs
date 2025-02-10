using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot.Data.TwelveDataAPI
{
    public class StockDataResponse
    {
        [JsonProperty("meta")]
        public MetaData Meta { get; set; }

        [JsonProperty("values")]
        public List<StockValue> Values { get; set; }
    }

    public class MetaData
    {
        [JsonProperty("symbol")]
        public string Symbol { get; set; }

        [JsonProperty("interval")]
        public string Interval { get; set; }

        [JsonProperty("currency")]
        public string Currency { get; set; }

        [JsonProperty("exchange_timezone")]
        public string ExchangeTimezone { get; set; }

        [JsonProperty("exchange")]
        public string Exchange { get; set; }

        [JsonProperty("mic_code")]
        public string MicCode { get; set; }

        [JsonProperty("type")]
        public string Type { get; set; }
    }

    public class StockValue
    {
        [JsonProperty("datetime")]
        public DateTime DateTime { get; set; }

        [JsonProperty("open")]
        public decimal Open { get; set; }

        [JsonProperty("high")]
        public decimal High { get; set; }

        [JsonProperty("low")]
        public decimal Low { get; set; }

        [JsonProperty("close")]
        public decimal Close { get; set; }

        [JsonProperty("volume")]
        public int Volume { get; set; }
    }

}
