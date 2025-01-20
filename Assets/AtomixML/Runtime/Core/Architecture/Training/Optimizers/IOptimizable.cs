using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IOptimizable
    {
        public NVector Parameters { get; set; }
    }
}
