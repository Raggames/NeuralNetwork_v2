using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public struct KMeanClusteringOutputData : IMLInOutData
    {
        public int ClassLabel { get; set; }
        public double Euclidian { get; set; }
    }
}
