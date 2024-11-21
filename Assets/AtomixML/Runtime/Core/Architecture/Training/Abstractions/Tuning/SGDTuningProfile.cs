using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Training
{
    [Serializable]
    public class SGDTuningProfile : ITuningProfile<IStochasticGradientDescentParameters>
    {
        [SerializeField] private StochasticGradientDescentParameters _lowerBound;
        [SerializeField] private StochasticGradientDescentParameters _upperBound;

        public IStochasticGradientDescentParameters LowerBound => _lowerBound;

        public IStochasticGradientDescentParameters UpperBound => _upperBound;
    }
}
