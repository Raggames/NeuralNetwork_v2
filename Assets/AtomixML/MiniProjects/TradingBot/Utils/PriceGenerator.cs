using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public static class PriceGenerator
    {
        /// <summary>
        /// Markets tend to cluster around the open price, with fewer extreme movements.
        /// This is modeled using a normal distribution centered on the Open price, but constrained within[Low, High].
        /// </summary>
        /// <param name="open"></param>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <returns></returns>
        public static decimal GenerateGaussianPrice(decimal open, decimal low, decimal high)
        {
            double mean = (double)open;
            double stdDev = ((double)(high - low)) / 4; // Adjustable spread

            // Box-Muller transform to generate normal distribution
            double u1 = 1.0 - MLRandom.Shared.NextDouble();
            double u2 = 1.0 - MLRandom.Shared.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            double price = mean + stdDev * randStdNormal;
            return Math.Max(low, Math.Min(high, (decimal)price)); // Ensure within range
        }

        /// <summary>
        /// If you expect an uptrend or downtrend, you can bias price generation:
        /// In an uptrend, more prices should be near High.
        /// In a downtrend, more prices should be near Low.
        /// </summary>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <param name="biasFactor"></param>
        /// <returns></returns>
        public static decimal GenerateBiasedPrice(decimal low, decimal high, double biasFactor = 0.7)
        {
            double rnd = Math.Pow(MLRandom.Shared.NextDouble(), biasFactor); // Bias towards high
            return low + (decimal)rnd * (high - low);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="open"></param>
        /// <param name="low"></param>
        /// <param name="high"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public static List<decimal> GenerateGaussianPriceBatch(decimal open, decimal low, decimal high, int count)
        {
            List<decimal> prices = new List<decimal>();

            for (int i = 0; i < count; i++)
            {
                // Box-Muller transform for normal distribution
                var price = GenerateGaussianPrice(open, low, high);
                prices.Add((decimal)price);
            }

            return prices;
        }

        public static List<decimal> GenerateBiasedPriceBatch(decimal low, decimal high, int count, double biasFactor = 0.7)
        {
            List<decimal> prices = new List<decimal>();

            for (int i = 0; i < count; i++)
            {
                // Box-Muller transform for normal distribution
                var price = GenerateBiasedPrice(low, high, biasFactor);
                prices.Add((decimal)price);
            }

            return prices;
        }
    }
}
