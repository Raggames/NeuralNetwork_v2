using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class DirectionalChange
    {
        public enum FractalType { None, Bullish, Bearish }

        public class Fractal
        {
            public int Index;
            public decimal Price;
            public FractalType Type;

            public Fractal(int index, decimal price, FractalType type)
            {
                Index = index;
                Price = price;
                Type = type;
            }
        }

        public static Fractal FindFractals(List<MarketData> periods, int currentIndex)
        {
            if (periods.Count < 5)
            {
                throw new ArgumentException("Not enough data points to detect fractals.");
            }

            for (int i = 2; i < currentIndex - 2; i++)
            {
                // Bullish Fractal (Highest High in Window)
                if (periods[i].High > periods[i - 1].High && periods[i].High > periods[i - 2].High &&
                    periods[i].High > periods[i + 1].High && periods[i].High > periods[i + 2].High)
                {
                    return new Fractal(i, periods[i].High, FractalType.Bullish);
                }

                // Bearish Fractal (Lowest Low in Window)
                if (periods[i].Low < periods[i - 1].Low && periods[i].Low < periods[i - 2].Low &&
                    periods[i].Low < periods[i + 1].Low && periods[i].Low < periods[i + 2].Low)
                {
                    return new Fractal(i, periods[i].Low, FractalType.Bearish);
                }
            }

            return null;
        }
    }
}
