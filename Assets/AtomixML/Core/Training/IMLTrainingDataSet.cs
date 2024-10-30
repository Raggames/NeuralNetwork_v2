using NUnit.Framework;
using System.Collections.Generic;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Abstraction for features only dataset
    /// </summary>
    /// <typeparam name="TFeature"></typeparam>
    public interface IMLTrainingDataSet<TFeature> where TFeature : IMLInputData
    {
        public TFeature[] Features { get; }
    }

    /// <summary>
    /// Abstraction for labelised dataset
    /// </summary>
    /// <typeparam name="TFeature"></typeparam>
    /// <typeparam name="TLabel"></typeparam>
    public interface IMLLabelisedTrainingDataSet<TFeature, TLabel> : IMLTrainingDataSet<TFeature> where TFeature : IMLInputData where TLabel : IMLOutputData
    {
        public TLabel[] Labels { get; }
    }
}
