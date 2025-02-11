using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /// <summary>
    /// Abstraction for a scoring function/utility that wil uses weights tensor from the trading bot entity to compute an overall score for decision making
    /// </summary>
    /// <typeparam name="TInput"></typeparam>
    /// <typeparam name="KOutput"></typeparam>
    public interface ITradingBotScoringFunction<TInput, KOutput>
    {
        /// <summary>
        /// Initial parameters of the entity for this function.
        /// This property is used to initialize the weights on the entity so it is required to create the correct number of weights here.
        /// </summary>
        public double[] InitialParameters { get; set; }

        public KOutput ComputeScore(TInput input, decimal currentPrice, ref int weightIndex);
    }

    public interface IMomentumIndicator<TInput, KOutput> : ITradingBotScoringFunction<TInput, KOutput>
    {

    }

    public interface ITendancyIndicator<TInput, KOutput> : ITradingBotScoringFunction<TInput, KOutput>
    {

    }

    public interface IOscillatorIndicator<TInput, KOutput> : ITradingBotScoringFunction<TInput, KOutput>
    {

    }

    public interface IVolatilityIndicator<TInput, KOutput> : ITradingBotScoringFunction<TInput, KOutput>
    {

    }

    public interface IVolumeIndicator<TInput, KOutput> : ITradingBotScoringFunction<TInput, KOutput>
    {

    }
}
