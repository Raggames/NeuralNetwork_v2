using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public enum PositionTypes
    {
        None = 0,

        /// <summary>
        /// Buy / achat
        /// </summary>
        Long_Buy = 1,

        /// <summary>
        /// Sell / vente à découvert
        /// </summary>
        Short_Sell = 2,
    }
}
