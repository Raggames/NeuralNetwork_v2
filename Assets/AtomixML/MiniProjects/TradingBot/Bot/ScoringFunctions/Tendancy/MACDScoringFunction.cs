﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class MACDScoringFunction 
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            var crt_macd = input.manager.macd.current;
            int crt_period_index = input.manager.currentMarketSamples.Count;

            var p_previous = input.manager.currentMarketSamples[Math.Max(crt_period_index - 6, 0)].Close;

            // normalizing price with a previous period
            var normalized_price = decimal.ToDouble((currentPrice - p_previous ) / p_previous);

            // score = normalized_price * w1 * e^(k1 * indicator)
            var score = normalized_price * input.Weights[weightIndex] * Math.Exp(decimal.ToDouble(crt_macd.Histogram) *  input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
