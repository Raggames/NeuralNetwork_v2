using Atom.MachineLearning.Core.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IGradientDescentOptimizable : IOptimizable
    {
        public Task<double> Score();
    }
}
