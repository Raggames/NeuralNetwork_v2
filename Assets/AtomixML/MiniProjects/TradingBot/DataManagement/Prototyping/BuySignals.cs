using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public enum BuySignals
    {
        None = 0,

        /// <summary>
        /// Buy / achat
        /// </summary>
        Long_Sell = 1,

        /// <summary>
        /// Sell / vente à découvert
        /// </summary>
        Short_Buy = 2,
    }
}
