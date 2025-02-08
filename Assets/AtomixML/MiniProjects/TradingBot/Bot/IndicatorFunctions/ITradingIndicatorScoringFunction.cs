using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public interface ITradingIndicatorScoringFunction<TInput, KOutput>
    {
        public int ParametersCount { get; }
        public KOutput ComputeScore(TInput input, decimal currentPrice, ref int weightIndex);
    }

    public interface IMomentumIndicator<TInput, KOutput> : ITradingIndicatorScoringFunction<TInput, KOutput>
    {

    }

    public interface ITendancyIndicator<TInput, KOutput> : ITradingIndicatorScoringFunction<TInput, KOutput>
    {

    }

    public interface IOscillatorIndicator<TInput, KOutput> : ITradingIndicatorScoringFunction<TInput, KOutput>
    {

    }

    public interface IVolatilityIndicator<TInput, KOutput> : ITradingIndicatorScoringFunction<TInput, KOutput>
    {

    }

    public interface IVolumeIndicator<TInput, KOutput> : ITradingIndicatorScoringFunction<TInput, KOutput>
    {

    }
}
