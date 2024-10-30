using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.AtomixML.Core.Training
{
    public class UnsupervisedClassificationVectorNDataSet<TInput> : IMLTrainingDataSet<TInput> where TInput : IMLInputData
    {
        public TInput[] Features { get; }
    }
}
