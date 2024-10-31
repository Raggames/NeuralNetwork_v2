using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public abstract class KMCTrainingSet : IMLTrainingDataSet<NVector>
    {
        public abstract NVector[] Features { get; }
    }
}
