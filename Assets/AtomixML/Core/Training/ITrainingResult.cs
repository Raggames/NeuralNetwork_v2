using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public interface ITrainingResult
    {
        public LossFunctions LossFunction { get; }
        public float Accuracy { get; }
    }
}
